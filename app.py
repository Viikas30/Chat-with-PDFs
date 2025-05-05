import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Set page configuration at the start!
st.set_page_config(page_title="Smart PDF Chat", page_icon="📄")

def get_file_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.name.endswith(".txt"):
            text += file.read().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    return FAISS.from_texts(texts=text_chunks, embedding=SentenceTransformerEmbeddings(model_name="all-distilroberta-v1"))

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f'<div class="user-message">{message.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message.content}</div>', unsafe_allow_html=True)

def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("📄 Chat with Your PDFs")
    st.subheader("Upload multiple PDFs & ask questions")
    
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("PDF or .txt files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Analyzing..."):
                raw_text = get_file_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
