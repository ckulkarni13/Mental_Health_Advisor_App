import os
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_ENVIRONMENT")
pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to Pinecone index
index_name = "mental-health-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_region),
    )
index = pc.Index(index_name)

# OpenAI Initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    """
    Generate embeddings for a given text using OpenAI's text-embedding-ada-002 model.
    """
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

def store_data_in_pinecone(file_name):
    """
    Store embeddings and metadata in Pinecone with progress tracking and retries.
    """
    # Load processed data
    data = pd.read_csv(file_name, sep="|")

    # Batch upsert with progress bar
    batch_size = 25  # Smaller batch size
    with tqdm(total=len(data), desc="Storing data in Pinecone", unit="rows") as pbar:
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            vectors = [
                (f"row-{idx}", generate_embedding(row["Context"]), {"response": row["Response"]})
                for idx, row in batch.iterrows()
            ]
            # Safe upsert with retries
            try:
                safe_upsert(index, vectors)
            except Exception as e:
                print(f"Batch failed at index {i}: {e}")
                continue
            pbar.update(len(batch))

    print("Data successfully stored in Pinecone.")
    
def safe_upsert(index, vectors, retries=3):
    for attempt in range(retries):
        try:
            index.upsert(vectors=vectors)
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    raise Exception("Failed to upsert after multiple attempts.")

# Call the function with the processed data file
store_data_in_pinecone("processed_data_final.txt")