import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)
if __name__ == "__main__":
    print("Going to ingest llamaindex documentaion")
    print("pinecone api key:", os.environ["PINECONE_API_KEY"])
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamindex-docs-temp",
        file_extractor={
            ".pdf": UnstructuredReader(),
            ".html": UnstructuredReader(),
            ".eml": UnstructuredReader(),
        },
    )
    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=100, chunk_overlap=20)
    llm = OpenAI(model="gpt-3.5-turbo")
    embed_model = OpenAIEmbedding(embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )
    pinecone_index = pc.Index("llamaindex-documentation-helper")
    vectorstore = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vectorstore)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("done")
