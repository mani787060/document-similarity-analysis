# ===>  we want to see document similarity through the 'google_gemini_api'

import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# 1. Simplified Client Initialization
# The SDK handles the versioning automatically now.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# IMPORTANT: Use the exact model name string
MODEL_ID = "gemini-embedding-001" 

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about the rohit sharma"

try:
    print(f"--- Connecting to Google Gemini ({MODEL_ID}) ---")
    
    # 2. Generate Embeddings for Documents
    # gemini-embedding-001 is a high-performance 3072-dimension model
    doc_response = client.models.embed_content(
        model=MODEL_ID,
        contents=documents,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    doc_embeddings = [item.values for item in doc_response.embeddings]

    # 3. Generate Embedding for the Query
    query_response = client.models.embed_content(
        model=MODEL_ID,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_embedding = query_response.embeddings[0].values

    # 4. Manual Similarity Math (Safe for your system)
    def calculate_similarity(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # 5. Rank and Print
    scores = [calculate_similarity(query_embedding, doc_vec) for doc_vec in doc_embeddings]
    best_index = np.argmax(scores)

    print(f"\n✅ SUCCESS!")
    print(f"Query: {query}")
    print(f"Match: {documents[best_index]}")
    print(f"Similarity Score: {scores[best_index]:.4f}")

except Exception as e:
    print(f"\n❌ Final Troubleshooting Step:")
    print(f"Error: {e}")
    print("\nIf you still see 404, run this command in your terminal:")
    print("pip install -U google-genai")