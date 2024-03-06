from pathlib import Path

import arxiv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI

database = None
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
papers_path = Path("./papers")
papers_path.mkdir(exist_ok=True, parents=True)


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
        chunks = splitter.split_documents(pages)
        if globals()["database"] is None:
            globals()["database"] = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
        else:
            globals()["database"].add_documents(chunks)

    @tool("Ask a question of the current vectorstore.")
    def query_database(query: str) -> str:
        """Query the vector database to discover information stored there."""
        retriever = globals()["database"].as_retriever(search_type="similarity", search_kwargs={"k": 4})
        rqa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0.5),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = rqa(query)
        # chain = load_qa_chain(llm=OpenAI(temperature=0.5), verbose=True)
        # docs = globals()["database"].similarity_search(query, k=5)
        # result = chain.run(input_documents=docs, question=query)
        # result = " ".join(
        #     [
        #         dedent(
        #             f"""
        #             ---\n
        #             Source: {doc.metadata['source']}\n
        #             Page: {doc.metadata['page']}\n
        #             ---\n
        #             {doc.page_content}\n
        #             """
        #         )
        #         for doc in docs
        #     ]
        # )
        return str(result)
