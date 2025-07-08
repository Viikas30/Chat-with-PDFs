import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
import os

# Set page configuration at the start!
st.set_page_config(page_title="Smart PDF Chat", page_icon="📄")

chat_css = """
<style>
    

    /* Header styling */
    h1, h2, h3 {
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Chat message container styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 20px;
        border-radius: 12px;
        background-color: #dfdfdf;
        max-height: 550px;
        overflow-y: auto;
        margin-bottom: 20px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Modern message bubble base */
    .message-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 15px;
        position: relative;
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* User message modern style */
    .user-message {
        background-color: #dcf8c6;
        color: #2e7d32;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Bot message modern style */
    .bot-message {
        background-color: #e3f2fd;
        color: #1565c0;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 10px 15px;
        border: 1px solid #ccc;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        font-size: 15px;
    }

    /* Button styling */
    .stButton > button {
        background-color: #26a69a;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        background-color: #2bbbad;
        transform: translateY(-2px);
    }

    /* Spinner styling */
    .stSpinner > div > div {
        color: #26a69a;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f8e9;
        border-right: 1px solid #dcdcdc;
        padding: 20px;
    }
</style>
"""



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
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        st.error("OpenRouter API key not found. Please set it in the .env file.")
        return None
    llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",  # Specify the DeepSeek model
    api_key=openrouter_api_key  # Use your OpenRouter API key
)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    if st.session_state.conversation:
        # Get response from the conversation chain
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Display chat history in the main chat area
        # Use a div for the chat container to apply scrolling and consistent styling
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f'<div class="message-bubble user-message">{message.content}</div>', unsafe_allow_html=True)
            else:  # Bot message
                st.markdown(f'<div class="message-bubble bot-message">{message.content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please upload your documents and click 'Process' first.")

def main():
    load_dotenv()
    
    st.markdown(chat_css, unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main application header and subheader
    st.header("📄 Chat with Your Documents")
    st.subheader("Upload multiple PDFs & TXT files and ask questions")

    # User input text box for questions
    user_question = st.text_input("Ask a question about your documents here:", key="user_input_question")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for document upload and processing
    with st.sidebar:
        st.subheader("Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF or .txt files here:",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        if st.button("Process Documents", key="process_button"):
            if uploaded_files:
                with st.spinner("Analyzing documents... This might take a moment."):
                    # Get raw text from uploaded files
                    raw_text = get_file_text(uploaded_files)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                
                if st.session_state.conversation:
                    st.success("Documents processed successfully! You can now ask questions.")
                else:
                    st.error("Failed to initialize conversation chain. Check API key.")
            else:
                st.warning("Please upload at least one PDF or TXT file to process.")

if __name__ == '__main__':
    main()
