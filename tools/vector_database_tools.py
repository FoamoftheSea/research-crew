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
    @tool("Store arxiv.org paper into a local vector database using URL")
    def store_arxiv_paper(link: str) -> None:
        """Uses arxiv API to download paper, then store it into a vector database"""
        paper_id = link.split("/")[-1].replace(".pdf", "")
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        filename = f"{paper.pdf_url.split('/')[-1]}.{paper.title.replace(' ', '_')}.pdf"
        paper.download_pdf(dirpath="./papers", filename=filename)
        loader = PyPDFLoader(f"./papers/{filename}")
        pages = loader.load_and_split()
        if globals()["database"] is None:
            globals()["database"] = FAISS.from_documents(pages, OpenAIEmbeddings(model="text-embedding-3-small"))
        else:
            globals()["database"].add_documents(pages)

    @tool("Search current research vector database for the 2 most similar vectors to the input query")
    def search_vector_store(query: str) -> str:
        """Query the vector database to discover information stored there."""
        docs = globals()["database"].similarity_search(query, k=2)
        result = " ".join(
            [
                dedent(
                    f"""
                    ---\n
                    Source: {doc.metadata['source']}\n
                    Page: {doc.metadata['page']}\n
                    ---\n
                    {doc.page_content}\n
                    """
                )
                for doc in docs
            ]
        )
        return result
