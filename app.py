import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text 
def get_text_chunks(raw_text):
    text_splitter=RecursiveCharacterTextSplitter(
         chunk_size=1000,
         chunk_overlap=200
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks


 def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vector_store.persist()

  
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with mutiple pdf",  page_icon=":books :")

    st.header("Chat with mutiple pdf :books:")
    st.text_input("Enter your question here", key="question")
    
    with st.sidebar:
     st.subheader("Your document")
     pdf_docs=st.file_uploader("Upload your PDF file here", type=["pdf"], key="pdf_file", accept_multiple_files=True)
     if st.button("Process PDF", key="process_pdf"):
      with st.spinner("Processing..."): 
         raw_text=get_pdf_text(pdf_docs)
        
         text_chunks=get_text_chunks(raw_text)
         
         get_vector_store(text_chunks)
         st.success("Vector store created successfully!")


         
         
if __name__ == "__main__":
  main()