# chat_with_pdfs_debug_fixed.py
import os
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from typing import Any, Dict, Optional, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document


# ----------------------------
# Helper utilities
# ----------------------------
def find_first_text(obj: Any, min_len: int = 1) -> Optional[str]:
    """
    Recursively search a nested structure (dicts/lists/objects) and return the
    first non-empty string value found (length >= min_len).
    Useful for extracting answers from different LangChain return shapes.
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        s = obj.strip()
        return s if len(s) >= min_len else None

    if isinstance(obj, dict):
        # common direct keys
        for key in ("answer", "output_text", "result", "output", "text", "generated_text", "response"):
            if key in obj and isinstance(obj[key], str) and len(obj[key].strip()) >= min_len:
                return obj[key].strip()
        # some chain outputs place content inside nested lists/dicts
        for v in obj.values():
            t = find_first_text(v, min_len=min_len)
            if t:
                return t

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            t = find_first_text(item, min_len=min_len)
            if t:
                return t

    # LLMResult-like objects
    if hasattr(obj, "generations"):
        try:
            gens = getattr(obj, "generations")
            return find_first_text(gens, min_len=min_len)
        except Exception:
            pass

    # Objects with 'text' attribute (some wrappers)
    if hasattr(obj, "text") and isinstance(getattr(obj, "text"), str):
        t = getattr(obj, "text").strip()
        if len(t) >= min_len:
            return t

    # objects with 'content' attribute (sometimes used)
    if hasattr(obj, "content") and isinstance(getattr(obj, "content"), str):
        t = getattr(obj, "content").strip()
        if len(t) >= min_len:
            return t

    return None


def get_pdf_text(pdf_files) -> Dict[str, str]:
    """Extract text safely from uploaded PDF files using pypdf."""
    texts: Dict[str, str] = {}
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
        except Exception as e:
            texts[name] = ""
            st.session_state.setdefault("pdf_parse_errors", {})[name] = str(e)
    return texts


def get_text_chunks(texts: Dict[str, str]) -> List[Document]:
    """Split texts into Document objects with metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs: List[Document] = []
    for fname, text in texts.items():
        if not text:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": fname, "chunk": i}))
    return docs


def get_vector_store(docs: List[Document], persist_directory: str = "chroma_db") -> Chroma:
    """Create or load a Chroma vector store persisted to disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="documents",
    )
    # Chroma auto-persists in newer versions; ignore explicit persist failures
    try:
        vector_store.persist()
    except Exception:
        pass
    return vector_store


def get_conversation_chain(vector_store: Chroma, hf_token: str, retriever_k: int = 4) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with HuggingFaceEndpoint LLM and return it.
    """
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Set it in your environment.")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.0,
        max_new_tokens=512,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return conv_chain


# ----------------------------
# Invocation helpers
# ----------------------------
def invoke_chain_safely(chain: Any, question: str, chat_history: List = None) -> Any:
    """
    Invoke the chain with defensive patterns while ensuring correct input shape.
    Returns the raw chain result (whatever type the chain returns) or raises the last exception.
    """
    chat_history = chat_history or []
    last_exc = None

    # Preferred: pass a dict with the expected keys.
    try:
        if hasattr(chain, "invoke"):
            # Many Runnable/Chain implementations support .invoke(input_dict)
            return chain.invoke({"question": question, "chat_history": chat_history})
    except Exception as e:
        last_exc = e

    # Try calling the chain like a callable (some LangChain versions)
    try:
        return chain({"question": question, "chat_history": chat_history})
    except Exception as e:
        last_exc = e

    # Fallback to .run (returns str typically) but wrap it into dict for consistency
    try:
        if hasattr(chain, "run"):
            run_out = chain.run(question)
            return {"answer": run_out}
    except Exception as e:
        last_exc = e

    # Last attempt: try invoke with alternative key name (rare)
    try:
        if hasattr(chain, "invoke"):
            return chain.invoke({"input": question})
    except Exception as e:
        last_exc = e

    # If nothing worked, raise the last exception so caller can debug
    raise last_exc


# ----------------------------
# Session and UI init
# ----------------------------
def init_session():
    if "conversation_chain" not in st.session_state:
        st.session_state["conversation_chain"] = None
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "pdf_parse_errors" not in st.session_state:
        st.session_state["pdf_parse_errors"] = {}
    if "last_raw_response" not in st.session_state:
        st.session_state["last_raw_response"] = None


# ----------------------------
# Main app
# ----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs (fixed)", page_icon=":books:")
    init_session()

    st.title("Chat with multiple PDFs — fixed invocation ✅")

    # Sidebar
    with st.sidebar:
        st.subheader("Upload & Settings")
        pdf_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_files",
        )

        retriever_k = st.number_input("Retriever k (docs to fetch)", min_value=1, max_value=10, value=4)
        show_verbose = st.checkbox("Show verbose debug info", value=False)
        show_raw_response = st.checkbox("Show raw response", value=False)
        show_sources = st.checkbox("Show sources when available", value=True)
        st.markdown("---")
        hf_token_set = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        st.write("HuggingFace token present:", hf_token_set)
        if not hf_token_set:
            st.warning("Set HUGGINGFACEHUB_API_TOKEN in your environment for LLM calls to work.")

        if st.button("Process PDF(s)", key="process_pdf"):
            if not pdf_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Extracting text and building vector store..."):
                    texts = get_pdf_text(pdf_files)
                    if st.session_state["pdf_parse_errors"]:
                        st.write("Some PDF parsing errors occurred (see session_state['pdf_parse_errors'])")
                    docs = get_text_chunks(texts)
                    if not docs:
                        st.error("No extractable text found in uploaded PDFs. Ensure PDFs contain selectable text (not scanned images).")
                    else:
                        try:
                            vector_store = get_vector_store(docs)
                            st.success("Vector store created.")
                            st.session_state["vector_store"] = vector_store
                        except Exception as e:
                            st.error("Failed to create vector store:")
                            st.write(e)
                            st.session_state["vector_store"] = None

                        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                        if st.session_state["vector_store"] is not None and hf_token:
                            try:
                                chain = get_conversation_chain(st.session_state["vector_store"], hf_token, retriever_k)
                                st.session_state["conversation_chain"] = chain
                                st.success("Conversation chain created.")
                            except Exception as e:
                                st.error("Failed to create conversation chain:")
                                st.write(e)
                                st.session_state["conversation_chain"] = None
                        else:
                            st.info("Vector store created. Provide HUGGINGFACEHUB_API_TOKEN to create chain.")

    # Main area
    if st.session_state["conversation_chain"] is None:
        st.info("Upload PDFs and press 'Process PDF(s)' in the sidebar to get started.")
        if st.session_state["vector_store"] is not None:
            st.write("Vector store exists in session — chain not created (missing token or error).")
    else:
        st.subheader("Ask questions (answers will be sourced from uploaded PDFs only)")
        question = st.text_input("Enter your question", key="question_input")
        ask = st.button("Ask", key="ask_button")

        if ask and question:
            chain = st.session_state["conversation_chain"]
            retriever = None
            if st.session_state.get("vector_store") is not None:
                try:
                    retriever = st.session_state["vector_store"].as_retriever()
                except Exception:
                    # fallback: vector_store may already be a retriever-like object
                    retriever = None

            # check retriever/debug retrieval
            if retriever is not None:
                try:
                    # Prefer new API if available; otherwise fallback to get_relevant_documents
                    if hasattr(retriever, "get_relevant_documents"):
                        retrieved_docs = retriever.get_relevant_documents(question)
                    elif hasattr(retriever, "invoke"):
                        # some retrievers accept invoke with {"query": ...}
                        raw = retriever.invoke({"query": question})
                        # try to extract docs if returned as list-like
                        if isinstance(raw, list):
                            retrieved_docs = raw
                        else:
                            retrieved_docs = []
                    else:
                        retrieved_docs = []
                    st.write(f"Retrieved {len(retrieved_docs)} documents (debug).")
                    if len(retrieved_docs) == 0:
                        st.warning("Retriever returned 0 documents — model won't have context. Try re-processing PDFs or changing chunking / retriever k.")
                    if show_verbose and retrieved_docs:
                        st.write("First retrieved snippet (truncated):")
                        st.write(retrieved_docs[0].page_content[:2000])
                except Exception as e:
                    st.write("Error during retrieval debug:", e)

            # Invoke chain safely (always try to pass question as key)
            raw_result = None
            invocation_error = None
            try:
                raw_result = invoke_chain_safely(chain, question, chat_history=st.session_state.get("chat_history", []))
            except Exception as e:
                invocation_error = e
                raw_result = None

            st.session_state["last_raw_response"] = raw_result

            # Extract answer robustly
            answer = None
            if raw_result is not None:
                if isinstance(raw_result, dict):
                    answer = (
                        raw_result.get("answer")
                        or raw_result.get("result")
                        or raw_result.get("output")
                        or raw_result.get("output_text")
                        or find_first_text(raw_result, min_len=3)
                    )
                elif isinstance(raw_result, str):
                    answer = raw_result
                else:
                    answer = find_first_text(raw_result, min_len=3)

            if not answer:
                st.error("No answer returned from the model (extraction failed).")
                if invocation_error:
                    st.subheader("Invocation error (caught):")
                    st.write(invocation_error)
                if show_raw_response:
                    st.subheader("Raw response (debug)")
                    try:
                        st.json(raw_result)
                    except Exception:
                        st.write(type(raw_result))
                        st.write(raw_result)
                else:
                    st.write("Enable 'Show raw response' in the sidebar to inspect the chain's raw output.")
                # Show retrieved docs for debug (again)
                if st.session_state.get("vector_store") is not None:
                    try:
                        retr = st.session_state["vector_store"].as_retriever()
                        if hasattr(retr, "get_relevant_documents"):
                            debug_retrieved = retr.get_relevant_documents(question)
                        else:
                            debug_retrieved = []
                        st.subheader(f"Debug: {len(debug_retrieved)} retrieved documents")
                        if debug_retrieved:
                            st.write("First retrieved snippet (truncated):")
                            st.write(debug_retrieved[0].page_content[:2000])
                    except Exception as e:
                        st.write("Error while retrieving debug docs:", e)
            else:
                st.write(answer)
                # append to local chat history
                st.session_state["chat_history"].append((question, answer))

                # show sources if present
                if show_sources and isinstance(raw_result, dict):
                    sources = raw_result.get("source_documents") or raw_result.get("sources")
                    if sources:
                        st.markdown("**Sources:**")
                        for d in sources:
                            md = getattr(d, "metadata", {}) or {}
                            src = md.get("source", "<unknown>")
                            chunk_idx = md.get("chunk")
                            st.write(f"- {src}" + (f" (chunk {chunk_idx})" if chunk_idx is not None else ""))

                if show_verbose:
                    st.subheader("Session debug info")
                    st.write("chat_history length:", len(st.session_state["chat_history"]))
                    st.write("last_raw_response type:", type(raw_result))
                    if show_raw_response:
                        st.write("last_raw_response content:")
                        try:
                            st.json(raw_result)
                        except Exception:
                            st.write(raw_result)

        # conversation history UI
        if st.session_state["chat_history"]:
            st.markdown("---")
            st.subheader("Conversation history (most recent first):")
            for user_msg, assistant_msg in reversed(st.session_state["chat_history"]):
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**Assistant:** {assistant_msg}")
                st.markdown("")

    # Footer help
    st.markdown("---")
    st.caption("Tip: If retrieval returns zero documents, PDFs may be scanned images (no selectable text). Consider OCR (Tesseract) or re-run with searchable PDFs. If chain throws Missing input keys, make sure the chain is invoked with {'question': question, 'chat_history': [...] }.")

if __name__ == "__main__":
    main()
