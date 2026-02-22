from dotenv import load_dotenv
import os
import sys

# Try imports
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.schema import Document
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("No HuggingFace token found.")
    sys.exit(1)

try:
    print("Initializing LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.0,
        max_new_tokens=512,
    )

    print("Creating dummy vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content="Technical Skills refer to the specialized knowledge...")]
    vector_store = Chroma.from_documents(docs, embeddings)

    print("Creating chain...")
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )

    print("Invoking chain (Attempt 1)...")
    try:
        res = conv_chain.invoke({"question": "What are technical skills?", "chat_history": []})
        print("Success Attempt 1:")
        print(res)
    except Exception as e:
        print(f"Error 1: {repr(e)}")

    print("Invoking chain (Attempt 2)...")
    try:
        res = conv_chain({"question": "What are technical skills?", "chat_history": []})
        print("Success Attempt 2:")
        print(res)
    except Exception as e:
        print(f"Error 2: {repr(e)}")

except Exception as e:
    print(f"General error: {repr(e)}")
