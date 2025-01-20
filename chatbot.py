from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Step 1: Load the WikiText-103 dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
train_data = ds["train"]

# Step 2: Load pre-trained DistilBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Step 3: Function to get embeddings for a text chunk
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Step 4: Generate embeddings for the first 100 documents (to keep it small)
embeddings = []
texts = []

for i in range(100):
    paragraph = train_data[i]['text']
    embedding = get_embedding(paragraph)
    embeddings.append(embedding)
    texts.append(paragraph)

# Convert embeddings list into a NumPy array
embeddings = np.array(embeddings)

# Step 5: Simple in-memory vector database for fast querying
class VectorDatabase:
    def __init__(self, embeddings, texts):
        self.embeddings = embeddings
        self.texts = texts

    def query(self, query, top_k=3):
        query_embedding = get_embedding(query)
        similarities = self.cosine_similarity(query_embedding, self.embeddings)
        most_similar_idx = similarities.argsort()[-top_k:][::-1]
        return [(self.texts[i], similarities[i]) for i in most_similar_idx]

    def cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        cosine_similarity = (A . B) / (||A|| * ||B||)
        """
        dot_product = np.dot(vec1, vec2.T)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2, axis=1)
        return dot_product / (norm_vec1 * norm_vec2)

# Step 6: Create the vector database
vector_db = VectorDatabase(embeddings, texts)

query = input("Query: ")
results = vector_db.query(query)

# Display the results
print("\nTop 3 relevant passages:\n")
for i, (text, score) in enumerate(results):
    print(f"{i+1}. Similarity Score: {score:.4f}")
    print(f"Passage: {text[:500]}...")  # Display first 500 characters of the passage
    print()
