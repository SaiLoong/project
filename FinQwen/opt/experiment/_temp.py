from ..tools.config import Config

path = Config.company_txt_path(cid="03c625c108ac0137f413dfd4136adb55c74b3805")

with open(path, "r") as file:
    text = file.read()

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = UnstructuredFileLoader(path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
