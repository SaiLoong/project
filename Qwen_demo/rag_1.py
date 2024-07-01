# -*- coding: utf-8 -*-
# @file rag_1.py
# @author zhangshilong
# @date 2024/6/29

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

file_path = "qa_10.txt"
loader = UnstructuredFileLoader(file_path, mode="elements")
docs = loader.load()
# languages是列表，存chroma会报错
for doc in docs:
    doc.metadata.pop("languages")

# 已经一行一个Document，不需要再切分了
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# split_docs = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="m3e-base")

persist_directory = "demo_database"
# 默认是L2距离，改为cosine
vectordb = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_directory,
                                 collection_metadata={"hnsw:space": "cosine"})
