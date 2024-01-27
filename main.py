import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, ServiceContext
from pinecone import Pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms import OpenAI
import streamlit as st


MESSAGES = "messages"

st.set_page_config(
    page_title="Chat with llamaIndex docs, powered by LlamaIndex",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with LlamaIndex docs")


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    load_dotenv()
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    pinecone_index = pc.Index("llamaindex-documentation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )
    return index


index = get_index()

if "chat_engine" not in st.session_state.keys():
    st.session_state["chat_engine"] = index.as_chat_engine(
        chat_mode="context", verbose=True
    )

if MESSAGES not in st.session_state.keys():
    st.session_state[MESSAGES] = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's open source python library",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state[MESSAGES].append({"role": "user", "content": prompt})

for message in st.session_state[MESSAGES]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state[MESSAGES][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
