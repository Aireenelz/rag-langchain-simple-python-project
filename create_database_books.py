from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
import tiktoken

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

DATA_PATH_BOOKS = "data/books"
CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents_books()
    calculate_tokens(documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents_books():
    loader = DirectoryLoader(DATA_PATH_BOOKS, glob="*.md")
    documents = loader.load()
    return documents

def calculate_tokens(documents: list[Document]):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = 0

    for document in documents:
        num_tokens = len(enc.encode(document.page_content))
        total_tokens += num_tokens
    print(f"Total tokens in all documents: {total_tokens}")

    cost_per_1000_tokens = 0.0001
    total_cost = (total_tokens / 1000) * cost_per_1000_tokens
    print(f"Estimated cost to embed all documents: ${total_cost:.6f}")

def split_text(documents: list[Document]):
    # Splitting the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 100,
        length_function = len,
        add_start_index = True
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    #print("CHUNKS: \n", chunks)
    print("====================================================")
    print("CHUNKS[10]: \n", document)
    print("PAGE_CONTENT: \n", document.page_content)
    print("METADATA: \n", document.metadata)
    print("====================================================")

    return chunks

def save_to_chroma(chunks: list[Document]):
    
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Create a new database from the documents
    try:
        # Create a new database from the documents
        print("Initializing Chroma database...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=CHROMA_PATH
        )
        print("***before persist***")
        db.persist()
        print("***after persist***")
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    except Exception as e:
        print(f"Error while creating the Chroma database: {e}")
    
if __name__ == "__main__":
    main()