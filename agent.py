from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool

load_dotenv()
llm = OpenAI()


def write_haiku_topic(topic: str) -> str:
    '''
    Write a haiku on the given topic
    '''
    return llm.complete(f"Write a haiku about {topic} ")

def count_characters(text: str) -> str:
    '''
    Counts the number of characters in a text
    '''
    return len(text) * 100

if __name__ == "__main__":
    print("**** Agent LlamaIndex *****")
    # print(write_haiku_topic("mangoes"))
    tool1 = FunctionTool.from_defaults(fn=write_haiku_topic, name="Write")
    tool2 = FunctionTool.from_defaults(fn=count_characters, name="Count")
    agent = ReActAgent.from_tools(tools=[tool1, tool2], llm=llm, verbose=True)

    
    res = agent.query("Write me a haiku about the game of tennis and could you also count characters in the haiku")
    print(res)
