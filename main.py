import os
from dotenv import load_dotenv
from llama_index.readers import SimpleWebPageReader
from llama_index import VectorStoreIndex

def main(url: str) -> None:
  documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
  # print(len(documents))
  index = VectorStoreIndex.from_documents(documents=documents)
  query_engine = index.as_query_engine()
  response = query_engine.query("What is LlamaIndex ?")
  print(response)

if __name__ == '__main__':
  load_dotenv()
  print("Hello World. Let's learn llama index")
  print("Open api key is : ", os.environ['OPENAI_API_KEY'])
  print('********')
  main(url='https://cbarkinozer.medium.com/an-overview-of-the-llamaindex-framework-9ee9db787d16')