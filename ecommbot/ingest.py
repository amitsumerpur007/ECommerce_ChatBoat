import os
import pandas as pd
import dotenv
from ecommbot.data_converter import dataconverter
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
dotenv.load_dotenv()

# Fetch environment variables
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_KEY_SPACE = os.getenv("ASTRA_DB_API_KEY_SPACE")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate environment variables
if not all([ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_API_KEY_SPACE, GOOGLE_API_KEY]):
    raise ValueError("One or more required environment variables are missing.")


# Initialize embeddings and vector store
embedding = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")

vstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name="chatbotecomm",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_API_KEY_SPACE
)

def ingestdata(status):
    vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_API_KEY_SPACE,
        )
    
    storage=status
    
    if storage==None:
        docs=dataconverter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

if __name__=='__main__':
    vstore,inserted_ids=ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
            print(f"* {res.page_content} [{res.metadata}]")