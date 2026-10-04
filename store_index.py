from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "medical"

# Load and split documents
def load_and_split_documents(data_path="data"):
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"✅ Loaded and split {len(texts)} chunks.")
    return texts

# Main function to create and store the index
def main():
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found, creating it...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Corresponds to the embedding model's dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',     # or 'gcp'
                region='us-east-1' # ensure this matches your .env or Pinecone dashboard
            )
        )
        print(f"✅ Index '{INDEX_NAME}' created.")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

    # Load documents and create embeddings
    texts = load_and_split_documents()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Populate the index
    print("Populating the Pinecone index with document embeddings...")
    PineconeVectorStore.from_documents(texts, embedding_model, index_name=INDEX_NAME)
    print(f"✅ Successfully populated the Pinecone index '{INDEX_NAME}'.")

if __name__ == "__main__":
    main()