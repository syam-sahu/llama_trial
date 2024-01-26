from dotenv import load_dotenv
import os
from llama_index import SimpleDirectoryReader
from llama_index.node_parse import SimpleNodeParser


load_dotenv()

if __name__ == '__main__':
    print("Going to ingest llamaindex documentaion")
    print("pinecone api key:", os.environ["PINECONE_API_KEY"])