# -*- coding: utf-8 -*-
# @file rag_2.py
# @author zhangshilong
# @date 2024/6/29

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

embedding_model = HuggingFaceEmbeddings(model_name="m3e-base")
persist_directory = "demo_database"
# 数据库创建时已设为使用cosine距离
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
retriever = vectordb.as_retriever()


class Qwen(LLM):
    # 必须提前声明，不然实例化时基类pydantic.v1.BaseModel的__setattr__方法会报错
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        response, history = self.model.chat(self.tokenizer, prompt, history=None)
        return response

    @property
    def _llm_type(self) -> str:
        return "Qwen"


llm = Qwen(model_path="Qwen-1_8B-Chat")

template = """使用提供的材料来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。
材料：
{context}
问题：
{input}
有用的回答：
"""
prompt_template = PromptTemplate(template=template)

stuff_documents_chain = create_stuff_documents_chain(llm, prompt_template)

retrieval_chain = create_retrieval_chain(retriever, stuff_documents_chain)

question = "问题3__e_f4_的答案是啥"
# 直接问LLM
llm.invoke(question)

# 使用RAG
result = retrieval_chain.invoke({"input": question})
