from typing import List

import arxiv
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer
from transformers.utils.generic import PaddingStrategy


class MistralEmbeddings(Embeddings):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self.tokenizer(texts, max_length=300, truncation=True, padding=PaddingStrategy.MAX_LENGTH)
        return embeddings.data["input_ids"]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embeddings = self.tokenizer(text, max_length=300, truncation=True, padding=PaddingStrategy.MAX_LENGTH)
        return embeddings.data["input_ids"]


class VectorDatabaseTools:
    def __init__(self):
        self.database = None
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-7B-Instruct-v0.2", pad_token="</s>")
        # self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", pad_token="</s>")

    @tool("Use arxiv.org URL to download a paper and store into a local vector database")
    def store_arxiv_paper(self, link: str) -> None:
        """Uses arxiv API to download paper, then store it into a vector database"""
        paper_id = link.split("/")[-1].replace(".pdf", "")
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        filename = paper_id + ".pdf"
        paper.download_pdf(dirpath="./papers", filename=filename)
        loader = PyPDFLoader(filename)
        pages = loader.load_and_split()
        if self.database is None:
            self.database = FAISS.from_documents(pages, self.tokenizer)
        else:
            self.database.add_texts(pages)

    @tool("Chat with the current set of downloaded research papers using queries")
    def chat_with_docs(self, query: str) -> str:
        """Query the vector database to discover information stored there."""
        docs = self.database.similarity_search(query, k=2)
        result = " ".join(
            [
                f"""
                ---\n
                Source: {doc.metadata['source']}\n
                Page: {doc.metadata['page']}\n
                ---\n
                {doc.page_content}\n
                """
                for doc in docs
            ]
        )
        print(result)
        return result
