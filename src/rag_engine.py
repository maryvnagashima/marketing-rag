import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class MarketingRAG:
    """
    Motor simples de RAG focado em marketing, usando:
    - LangChain
    - ChromaDB como vector store
    - HuggingFaceEmbeddings com modelo open-source gratuito
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def build_knowledge_base(self, texts: List[str], metadatas: List[dict] = None) -> None:
        """
        Cria a base vetorial a partir de uma lista de textos.

        Parameters
        ----------
        texts : List[str]
            Lista de documentos em texto.
        metadatas : List[dict], optional
            Metadados associados a cada documento.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )

        docs: List[Document] = []
        for text, meta in zip(texts, metadatas):
            for chunk in splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata=meta))

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def load_existing_knowledge_base(self) -> None:
        """
        Carrega uma base já persistida no disco.
        """
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Recupera os k documentos mais relevantes para a query.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore não inicializado. Chame build_knowledge_base ou load_existing_knowledge_base primeiro.")
        return self.vectorstore.similarity_search(query, k=k)

    def answer(self, query: str, k: int = 4) -> str:
        """
        Gera uma resposta simples concatenando trechos recuperados.
        (Focado em demonstrar o pipeline RAG; o LLM pode ser plugado depois.)
        """
        docs = self.retrieve(query, k=k)
        context = "\n---\n".join(d.page_content for d in docs)
        answer = (
            f"Pergunta: {query}\n\n"
            f"Contexto relevante encontrado:\n{context}\n\n"
            "Resumo: com base nos trechos acima, analise manualmente o melhor canal, "
            "campanha ou insight de marketing."
        )
        return answer
