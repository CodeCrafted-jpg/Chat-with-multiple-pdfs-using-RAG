# Chat with PDFs - App Workflow Architecture

This document outlines the detailed step-by-step workflow, data pipeline, and architecture of the `app.py` Streamlit application.

## 1. Init Phase: Application & Session Setup
When the Streamlit app loads, the `main()` function handles the initial setup:
- **Environment**: `load_dotenv()` runs to load environment variables, specifically searching for `HUGGINGFACEHUB_API_TOKEN` which is crucial for authenticating the LLM.
- **UI Config**: Sets the page title to "Chat with multiple PDFs" and configures the icon.
- **Session State Initialization (`init_session`)**: Initializes critical variables in `st.session_state` to retain data across app reruns:
  - `conversation_chain`: Stores the LangChain pipeline once created.
  - `vector_store`: Stores the Chroma vector database instance.
  - `chat_history`: Keeps track of prior user queries and assistant responses.
  - `pdf_parse_errors`: Logs dictionary for any pages that failed extraction.
  - `last_raw_response`: Stores raw AI responses for debugging purposes.

## 2. Ingestion Phase: PDF Upload & Processing
The sidebar provides an interface for users to upload files and configure settings. Processing triggers when the user clicks **"Process PDF(s)"**:

1. **Text Extraction (`get_pdf_text`)**:
   - The app reads binary contents of all uploaded PDFs.
   - Using `pypdf.PdfReader`, it iterates through every page and extracts the text.
   - If an error occurs on a specific PDF, it is caught safely and logged into the session state without crashing the app.
2. **Text Splitting/Chunking (`get_text_chunks`)**:
   - The raw text is passed to LangChain's `RecursiveCharacterTextSplitter`.
   - The text is broken down into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`). This ensures sentences aren't cut off abruptly and context is preserved.
   - The chunks are wrapped in LangChain `Document` objects with metadata tracking their source filename and sequence index.
3. **Vector Store Generation (`get_vector_store`)**:
   - The app uses `HuggingFaceEmbeddings` pointing to the `"sentence-transformers/all-MiniLM-L6-v2"` model to convert text chunks into dense numeric vectors (embeddings).
   - A local `Chroma` vector database is instantiated, saving the embeddings persistently to the `./chroma_db/` folder on disk.
4. **Conversational Chain Setup (`get_conversation_chain`)**:
   - With a populated vector store, a language model is initialized via `HuggingFaceEndpoint` pointing to `"mistralai/Mistral-7B-Instruct-v0.2"`. Temperature is `0.0` for consistent, non-hallucinated answers.
   - A `ConversationalRetrievalChain` is constructed tying together the LLM and the Vector Store Retriever (fetching top `k` docs), effectively equipping the LLM with chat history memory and context-aware context retrieval.

## 3. Query Phase: Asking Questions
The main chat UI activates once the pipeline is built. When the user submits a question and clicks **"Ask"**:

1. **Security & Validation Retrieval**:
   - The app performs a discrete retrieval test, pulling relevant documents from the vector store purely to print debug bounds and warn the user if no relevant chunks are found.
2. **Execution / LLM Invocation (`invoke_chain_safely`)**:
   - The question and current `chat_history` are submitted to the conversation chain.
   - **Fault-Tolerance Mechanism**: Because LangChain rapidly changes its API methods, `invoke_chain_safely` attempts calling the chain using several different method syntaxes (`.invoke()`, `__call__()`, `.run()`, etc.) ensuring cross-version compatibility.
3. **Robust Answer Extraction (`find_first_text`)**:
   - The response from LangChain chains can be shaped differently depending on its config. `find_first_text` is a recursive utility that tunnels through nested dictionaries, lists, and generator objects to find the primary semantic string output authored by the LLM.
4. **UI Render and State Update**:
   - The extracted answer is printed to the screen.
   - The `(question, answer)` tuple is appended to `st.session_state["chat_history"]`.
   - If toggled, metadata sources (document name & chunk ID) are displayed.
5. **Chat History Display**:
   - The UI runs through the updated list of `chat_history` backwards, rendering the user-assistant thread with the newest messages up top.
