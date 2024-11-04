import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["pcsk_6kZfxW_67jSYpsh3GBo1HGtijStRMttkV6VmDPVapV1kgniysTobpzfUWQ6da8psY4JoMB"]
pinecone_environment = os.environ["us-east-1"] 
index_name = os.environ["langchainvector"]  

# Create Streamlit app
st.set_page_config(page_title="Ask your PDFs")
st.header("Ask your PDFs")

# Upload multiple PDF files
pdfs = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

# Extract text from multiple PDFs
if pdfs is not None and len(pdfs) > 0:
    text_data = []
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text_data.append(page.extract_text())

    # Preprocess text data
    def read_doc(text_data):
        return text_data

    doc = read_doc(text_data)

    # Divide the docs into chunks
    def chunk_data(docs, chunk_size=800, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = text_splitter.split_documents(docs)
        return doc

    documents = chunk_data(docs=doc)

    # Embedding Technique Of OPENAI
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Vector Search DB In Pinecone
    Pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )
    index = Pinecone.from_documents(documents, embeddings, index_name=index_name)  # Changed doc to documents

    # Cosine Similarity Retrieve Results from VectorDB
    def retrieve_query(query, k=2):
        matching_results = index.similarity_search(query, k=k)
        return matching_results

    # Function to ask MedQA
    def medqa_ask(question, context):
        # Placeholder for MedQA integration
        # Replace this with the actual call to your MedQA model
        # This could be an API call or a local model call.
        return "This is a placeholder response from MedQA."

    # Search answers from VectorDB
    def retrieve_answers(query):
        doc_search = retrieve_query(query)
        context = "\n".join([doc.page_content for doc in doc_search])  # Assuming doc_search has a page_content attribute
        response = medqa_ask(query, context)
        return response

    # Show user input for the question
    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        answer = retrieve_answers(user_question)
        st.write(answer)