import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, ServiceContext
from pinecone import Pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.llms import OpenAI
from llama_index.postprocessor import SentenceEmbeddingOptimizer
import streamlit as st
from node_postprocessors import DuplicateNodeRemoverPostprocessor


MESSAGES = "messages"


llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

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

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )
    return index


index = get_index()

if "chat_engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )

    st.session_state["chat_engine"] = index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        node_postprocessors=[postprocessor, DuplicateNodeRemoverPostprocessor()],
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
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source node { i+1 } score = {node.score}")
                    st.write(node.text)
            st.session_state[MESSAGES].append(
                {"role": "assistant", "content": response.response}
            )
