
import os
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint


def get_pdf_text(pdf_files):
    """Extract text safely from uploaded PDF files using pypdf.

    Returns a dict mapping filename -> extracted text.
    """
    texts = {}
    for pdf in pdf_files or []:
        name = getattr(pdf, "name", "uploaded_pdf")
        try:
            reader = PdfReader(pdf)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            texts[name] = "\n".join(pages)
        except Exception:
            # skip files that fail to parse
            texts[name] = ""
    return texts


def get_text_chunks(texts: dict):
    """Split texts into Document objects with metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for fname, text in texts.items():
        if not text:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": fname, "chunk": i}))
    return docs


def get_vector_store(docs):
    """Create or load a Chroma vector store persisted to ./chroma_db."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="documents",
    )
    try:
        # ensure the DB is written to disk
        vector_store.persist()
    except Exception:
        pass
    return vector_store


def get_conversation_chain(vector_store):
    """Build a retrieval chain using HuggingFaceEndpoint and the provided vector_store.

    The prompt instructs the model to answer only from context and to say it doesn't know
    when the answer isn't present in the context.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.0,
        max_new_tokens=512,
    )

    retriever = vector_store.as_retriever()

    system_message = (
        "You are a concise and professional assistant. Answer using ONLY the provided context."
        " If the answer is not in the context, say you don't know."
    )

    system = SystemMessagePromptTemplate.from_template(system_message)
    human_template = "Context:\n{context}\n\nQuestion: {input}\n\nAnswer concisely and professionally."
    human = HumanMessagePromptTemplate.from_template(human_template)

    prompt = ChatPromptTemplate.from_messages([system, human])

    # create the stuff (combine documents) chain and the retrieval wrapper
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def init_session():
    if "conversation_chain" not in st.session_state:
        st.session_state["conversation_chain"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    init_session()

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_files",
        )

        if st.button("Process PDF", key="process_pdf"):
            if not pdf_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs and creating vectors..."):
                    texts = get_pdf_text(pdf_files)
                    docs = get_text_chunks(texts)
                    if not docs:
                        st.error("No extractable text found in uploaded PDFs.")
                    else:
                        vector_store = get_vector_store(docs)
                        st.success("Vector store created successfully!")
                        # build and store conversation chain in session state
                        chain = get_conversation_chain(vector_store)
                        st.session_state["conversation_chain"] = chain
                        # keep the vector store so we can inspect retrieved docs for debugging
                        st.session_state["vector_store"] = vector_store

    # Main chat area
    if st.session_state["conversation_chain"] is None:
        st.info("Upload and process PDFs from the sidebar to start chatting.")
    else:
        question = st.text_input("Enter your question here", key="question_input")
        if st.button("Ask", key="ask_button") and question:
            chain = st.session_state["conversation_chain"]

            # Attempt multiple invocation patterns to support different LangChain Runnable APIs
            response = None
            for payload in ({"query": question}, {"input": question}, question):
                try:
                    if hasattr(chain, "invoke"):
                        response = chain.invoke(payload)
                    else:
                        response = chain(payload)
                    break
                except Exception:
                    response = None
                    continue

            answer = None
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("output_text") or response.get("result") or response.get("output")
            elif isinstance(response, str):
                answer = response

            if not answer:
                st.error("No answer returned from the model.")
                # show raw response for debugging
                st.subheader("Debug: raw response")
                st.write(response)

                # if we have the vector store, show what documents were retrieved
                if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
                    try:
                        retriever = st.session_state["vector_store"].as_retriever()
                        retrieved = retriever.get_relevant_documents(question)
                        st.subheader(f"Debug: {len(retrieved)} retrieved documents")
                        if retrieved:
                            # show the first retrieved doc (truncated)
                            snippet = retrieved[0].page_content
                            st.write(snippet[:2000])
                    except Exception as e:
                        st.write("Error while retrieving debug docs:", e)
            else:
                st.write(answer)


if __name__ == "__main__":
    main()