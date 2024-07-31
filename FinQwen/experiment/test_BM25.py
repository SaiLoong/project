# -*- coding: utf-8 -*-
# @file test_BM25.py
# @author zhangshilong
# @date 2024/7/19

import jieba
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..tools.config import Config

questions = [
    "青海互助青稞酒股份有限公司的主营业务是什么？",
    "青海互助青稞酒股份有限公司主要困难是什么？",
    "青海互助青稞酒股份有限公司的实际控制人是谁？共有几人？",
    "青海互助青稞酒股份有限公司报告期内面临的最重要风险因素是什么？",
    "青海互助青稞酒股份有限公司设置哪些部门？",
    "青海互助青稞酒股份有限公司2010年经备案的青稞收购量及实际采购量是多少？",
    "青海互助青稞酒股份有限公司2010年度营业收入与净利润是多少？",
    "青海互助青稞酒股份有限公司材料采购物资主要有哪些？",
    "青海互助青稞酒股份有限公司本次发行股票的数量、每股面值分别是多少？",
    "青海互助青稞酒股份有限公司本次发行后总股本不超过多少？"
]
company = "青海互助青稞酒股份有限公司"
path = Config.company_txt_path(company=company)

loader = UnstructuredFileLoader(path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)


# 先保留标点符号
def preprocessing_func(text):
    return jieba.lcut(text)


# 默认k=4
retriever = BM25Retriever.from_documents(split_docs, preprocess_func=preprocessing_func, k=4)

question = questions[0]
print(question, "\n\n")
result = retriever.invoke(question)
for doc in result:
    print(repr(doc.page_content), "\n")

"""
试了几个问题，BM25的效果不是很好，应该与停用词无关，是关键词也比较通用，容易匹配到别的材料

“主要困难”散落在子标题中，难以检索
貌似问题的确能在标题中寻找
"""
