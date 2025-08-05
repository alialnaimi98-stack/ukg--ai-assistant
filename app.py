import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.agents import initialize_agent, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_types import AgentType
import os
import tempfile

# ---- Page Setup ----
st.set_page_config(page_title="UKG WFM AI Assistant", layout="wide")
st.title("UKG WFM AI Assistant (Prototype)")

# ---- File Upload UI ----
st.markdown("### 📤 Upload UKG Training Slides (PDF or TXT)")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True)

all_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        all_docs.extend(loader.load())

# ---- Split Documents ----
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)

# ---- Embedding & Vector Store ----
if split_docs:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)
else:
    qa_chain = None

# ---- Internet Search Tool ----
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for answering questions from the internet or UKG community"
    )
]
agent = initialize_agent(
    tools, ChatOpenAI(model_name="gpt-3.5-turbo"), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)

# ---- Query Input ----
query = st.text_input("Ask a question about UKG WFM:")

if query:
    response_sources = []

    # 1. Try internal documents
    if qa_chain:
        with st.spinner("Checking internal documents..."):
            result = qa_chain({"question": query})
        st.markdown("#### 📄 Answer from Internal Docs:")
        st.success(result['answer'])
        if result.get('sources'):
            response_sources.append(f"Internal: {result['sources']}")
            st.caption(f"Sources: {result['sources']}")
    else:
        st.info("No internal files loaded.")

    # 2. Internet Search
    with st.spinner("Searching the internet (UKG Community)..."):
        web_result = agent.run(query)
    st.markdown("#### 🌐 Internet Search Answer:")
    st.info(web_result)
    response_sources.append("Web: Google Search")

    # 3. GPT General Answer
    with st.spinner("Asking ChatGPT directly..."):
        direct_response = ChatOpenAI(model_name="gpt-3.5-turbo").predict(query)
    st.markdown("#### 💬 ChatGPT's Answer:")
    st.info(direct_response)

st.markdown("---")
st.markdown("This prototype combines internal training slides, internet search, and ChatGPT to give complete answers to your UKG WFM questions.")
