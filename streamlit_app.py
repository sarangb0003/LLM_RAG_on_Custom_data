import streamlit as st
from streamlit_chat import message
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader,TextLoader
# from langchain.vectorstores import FAISS
#from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import Chroma
import tempfile

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
repo_id = "google/flan-t5-xxl"

# App title
st.set_page_config(page_title="LLM Chatbot")

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

# Hugging Face Credentials
if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(file_path=tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    

    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=user_api_key)
    vectors = Chroma.from_documents(docs, embeddings)
    
    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    

    chain = ConversationalRetrievalChain.from_llm(llm = HuggingFaceHub(huggingfacehub_api_token=user_api_key,
                                                                       repo_id=repo_id, retriever=retriever),
                                                                      return_source_documents=True,
                                                                        return_generated_question=True,model_kwargs={"temperature": 0.1})
    def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
