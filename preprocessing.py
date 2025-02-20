from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import time  # For timing
# import cupy as cp

import genai.lama  # GPU-accelerated computation
# cp.cuda.Device(0).use()
import streamlit as st
import pickle
import genai
import torch

# def chunk_text(text, chunk_size=500, overlap=100):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunks.append(" ".join(words[i:i + chunk_size]))
#     return chunks

# Function to get relevant chunks using TF-IDF and cosine similarity
def get_relevant_chunks_re(query, chunks, top_n=1):
    """ Retrieve the top-N relevant chunks based on cosine similarity. """
    if not query or not chunks:
        return []  # Early exit if query or chunks are empty

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform query and chunks into TF-IDF vectors
    vectorizer.fit(chunks)
    query_vector = torch.tensor(vectorizer.transform([query]).toarray()).float()
    chunks_vectors = torch.tensor(vectorizer.transform(chunks).toarray()).float()

    # Compute cosine similarity between the query and chunk vectors
    cosine_sim = torch.matmul(query_vector, chunks_vectors.T) / (
        torch.norm(query_vector) * torch.norm(chunks_vectors, dim=1)
    )

    # Convert cosine similarity to numpy for sorting
    cosine_sim = cosine_sim.cpu().numpy()  # Move back to CPU for compatibility

    # Get top-N relevant indices based on cosine similarity
    relevant_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    print("Relevant Indices:", relevant_indices)
    
    return [chunks[i] for i in relevant_indices] if relevant_indices.size else []

# Function to chunk text
# import re
# def chunk_text(text, chunk_size=500, overlap=100):
#     # words = text.split()
#     words =re.findall(r'\S+', text)
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunks.append(" ".join(words[i:i + chunk_size]))
#     return chunks

# Function to parallelize chunking
import re
import numpy as np
def chunk_text_by_special_character(text, separator="---"):
    """Splits text into chunks based on the specified separator."""
    chunks = re.split(rf'\s*{re.escape(separator)}\s*', text)
    chunks = [str(chunk).lower() for chunk in chunks]
    # print(chunks)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# Define the chunking function
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Parallel chunking function
def parallel_chunk_texts(texts, chunk_size=500, overlap=100):
    chunks = []
    
    # Debugging: Print input texts to verify it's as expected
    print(f"Input texts: {texts}")
    
    with ThreadPoolExecutor() as executor:
        # Executor map applies chunk_text to each item in texts
        results = executor.map(chunk_text, texts, [chunk_size] * len(texts), [overlap] * len(texts))
        
        # Debugging: Print the results of each thread execution
        print(f"Results from executor: {list(results)}")
        
        for result in results:
            chunks.extend(result)
        
        # Display chunks in Streamlit
        st.write(chunks)
    
    return chunks


# Function to get relevant chunks using TF-IDF and cosine similarity
# import torch

# # import pickle
# import torch
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer

# VECTOR_DIR = "..dbs/"  # Directory to store vector files
# VECTOR_FILE =  "./dbs/tfidf_vectors.pkl"

# def save_vectors(vectorizer, vectors, chunks):
#     """ Save vectorizer, vectors, and chunks to a pickle file. """
#     os.makedirs(VECTOR_DIR, exist_ok=True)
#     with open(VECTOR_FILE, "wb") as f:
#         pickle.dump((vectorizer, vectors, chunks), f)

# def load_vectors():
#     """ Load vectorizer, vectors, and chunks from a pickle file. """
#     if os.path.exists(VECTOR_FILE):
#         with open(VECTOR_FILE, "rb") as f:
#             return pickle.load(f)  # Returns (vectorizer, vectors, chunks)
#     return None

# def get_relevant_chunks(query, chunks, top_n=1,previous=None):
#     """ Retrieve top relevant chunks using stored TF-IDF vectors. """
#     # question =genai.lama.query_lm_studio(query,previous+"these are the previous question if the current question is not understandable or it may have possible to follow back question generate the correct question or return the user question")
#     # Try to load existing vectors
#     data = load_vectors()
    
#     if data:
#         vectorizer, vectors, saved_chunks = data
#         # If chunks are different, recompute vectors
#         if saved_chunks != chunks:
#             print("Chunks updated, recomputing vectors...")
#             vectorizer = TfidfVectorizer()
#             vectors = vectorizer.fit_transform(chunks)
#             save_vectors(vectorizer, vectors, chunks)  # Save updated vectors
#     else:
#         print("No saved vectors found, computing for the first time...")
#         vectorizer = TfidfVectorizer()
#         vectors = vectorizer.fit_transform(chunks)
#         save_vectors(vectorizer, vectors, chunks)  # Save vectors


#     # Convert to PyTorch tensors and move to GPU
#     vectors_gpu = torch.tensor(vectors.toarray(), device="cpu")
    
#     # Compute query vector separately
#     query_vector = torch.tensor(vectorizer.transform([query]).toarray(), device="cpu").float()
#     chunks_vectors = vectors_gpu.float()

#     # Compute cosine similarity using PyTorch
#     cosine_sim = torch.matmul(query_vector, chunks_vectors.T) / (
#         torch.norm(query_vector) * torch.norm(chunks_vectors, dim=1)
#     )
#     cosine_sim = cosine_sim.cpu().numpy()  # Move back to CPU for compatibility
#     st.text(f"similarity\n{cosine_sim}")
#     # Get top-N relevant chunks
#     # relevant_indices = cosine_sim[0].argsort()[-top_n:][::-1]
#     threshold = 0.2
#     cosine_sim_row = cosine_sim[0]
#     print("Shape of cosine_sim:", cosine_sim.shape)
#     # print("cosine_sim array:", cosine_sim)
# # Get indices where similarity is greater than the threshold
#     filtered_indices = np.where(cosine_sim_row > threshold)[0]
#     print(filtered_indices)
#     # Sort the filtered indices based on similarity scores in descending order
#     if len(filtered_indices)>0:
#         relevant_indices = filtered_indices[np.argsort(cosine_sim_row[filtered_indices])[::-1]]
#     else:
#         relevant_indices =[]
#     print("Relevant Indices:", relevant_indices)
#     return [chunks[relevant_indices[0]]]
#     # return [chunks[i] for i in relevant_indices]
import torch
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "../dbs/"  # Directory to store vector files
VECTOR_FILE = "./dbs/tfidf_vectors.pkl"

# Load the sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

def save_vectors(vectorizer, vectors, chunks):
    """ Save vectorizer, vectors, and chunks to a pickle file. """
    os.makedirs(VECTOR_DIR, exist_ok=True)
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump((vectorizer, vectors, chunks), f)

def load_vectors():
    """ Load vectorizer, vectors, and chunks from a pickle file. """
    if os.path.exists(VECTOR_FILE):
        with open(VECTOR_FILE, "rb") as f:
            return pickle.load(f)  # Returns (vectorizer, vectors, chunks)
    return None

# def get_relevant_chunks(query, chunks, top_n=1, previous=None):
#     """ Retrieve top relevant chunks using stored TF-IDF vectors. """

#     # Try to load existing vectors
#     data = load_vectors()
    
#     if data:
#         vectorizer, vectors, saved_chunks = data
#         # If chunks are different, recompute vectors
#         if saved_chunks != chunks:
#             print("Chunks updated, recomputing vectors...")
#             vectorizer = TfidfVectorizer()
#             vectors = vectorizer.fit_transform(chunks)
#             save_vectors(vectorizer, vectors, chunks)  # Save updated vectors
#     else:
#         print("No saved vectors found, computing for the first time...")
#         vectorizer = TfidfVectorizer()
#         vectors = vectorizer.fit_transform(chunks)
#         save_vectors(vectorizer, vectors, chunks)  # Save vectors

#     # Convert text chunks to embeddings
#     chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

#     # Compute query embedding
#     query_embedding = model.encode(query, convert_to_tensor=True)

#     # Compute cosine similarity using semantic embeddings
#     cosine_sim = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings, dim=1)
    
#     # Convert to NumPy for easy indexing
#     cosine_sim_np = cosine_sim.cpu().numpy()
#     st.text(cosine_sim_np)
#     # Apply threshold
#     threshold = 0.2
#     filtered_indices = np.where(cosine_sim_np > threshold)[0]

#     print("Shape of cosine_sim:", cosine_sim_np.shape)
#     print("Filtered indices:", filtered_indices)

#     # Sort indices based on similarity scores in descending order
#     if len(filtered_indices) > 0:
#         relevant_indices = filtered_indices[np.argsort(cosine_sim_np[filtered_indices])[::-1]]
#     else:
#         relevant_indices = []
    
#     print("Relevant Indices:", relevant_indices)
    

#     return [chunks[relevant_indices[0]]] if relevant_indices.size else []

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def get_relevant_chunks(query, chunks, top_n=1, previous=None):
    """ Retrieve top relevant chunks using stored TF-IDF vectors. """

    # Generate refined question
#     refined_question = genai.lama.query_question_maker(
#     prompt=f"Current question: {query}", 
#     context=f"try to match the Current question with These are the previous questions: {previous}. If the current question is unclear or follows up on a previous one, generate a more precise version of the question. Otherwise, return the original question as it is.\n\n"
#             f"Return only the refined question without any explanation or additional context."
# )
    refined_question=None

    
    # Load stored vectors if available
    data = load_vectors()
    
    if data:
        vectorizer, vectors, saved_chunks = data
        if saved_chunks != chunks:
            print("Chunks updated, recomputing vectors...")
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(chunks)
            save_vectors(vectorizer, vectors, chunks)  # Save updated vectors
    else:
        print("No saved vectors found, computing for the first time...")
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(chunks)
        save_vectors(vectorizer, vectors, chunks)  # Save vectors

    # Convert vectors to NumPy array before sending to GPU
    vectors_array = vectors.toarray()
    vectors_gpu = torch.tensor(vectors_array, device="cpu")

    # Choose query or refined question based on similarity
    # refined_vector = vectorizer.transform([refined_question]).toarray()
    # original_vector = vectorizer.transform([query]).toarray()

    # refined_sim = np.dot(refined_vector, original_vector.T)[0][0]  # Cosine similarity
    
    # if refined_sim > 0.8:  # Use refined question if similarity is high
    #     final_query = refined_question
    # else:
    final_query = query

    print(f"Using Query: {final_query}")

    # Compute final query vector and move to GPU
    query_vector = torch.tensor(vectorizer.transform([final_query]).toarray(), device="cpu").float()
    chunks_vectors = vectors_gpu.float()

    # Compute cosine similarity using PyTorch
    cosine_sim = torch.matmul(query_vector, chunks_vectors.T) / (
        torch.norm(query_vector) * torch.norm(chunks_vectors, dim=1)
    )
    cosine_sim = cosine_sim.cpu().numpy()  # Move back to CPU for compatibility

    # Get top-N relevant chunks
    relevant_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    print("Relevant Indices:", relevant_indices)
    
    return [chunks[i] for i in relevant_indices]


def create_combined_prompt(question, sql_prompt):
    return f"{sql_prompt}\n\n{question}\n\n"


responses = {
        "current department": ["my", "our"],
        "UGTAMIL": ["tamil department ug", "department of tamil", "bsc tamil", "ug tamil","tamil department"],
        "UGHINDI": ["hindi department ug", "department of hindi", "ba hindi", "ug hindi","hindi department"],
        "PART2ENG": ["english department ug", "department of english", "ba english", "ug english","english department"],
        "UGMAT": ["mathematics department ug", "department of mathematics", "bsc mathematics", "ug mathematics","maths department"],
        "UGPHY": ["physics department ug", "department of physics", "bsc physics", "ug physics","phusics department"],
        "UGCHE": ["chemistry department ug", "department of chemistry", "bsc chemistry", "ug chemistry","chemistry department"],
        "UGBOT": ["botany department ug", "department of botany", "bsc botany", "ug botany","botany department"],
        "UGZOO": ["zoology department ug", "department of zoology", "bsc zoology", "ug zoology","zoology department"],
        "UGPHS": ["physical education department ug", "department of physical education", "bsc physical education", "ug physical education","phs","physical education department"],
        "UGECO": ["economics department ug", "department of economics", "ba economics", "ug economics","economic department","economic department"],
        "UGCOM": ["commerce department ug", "department of commerce", "bcom commerce", "ug commerce","bcom","commerce general","commerce department"],
        "UGBBAR": ["business administration department ug", "department of business administration", "bba business administration", "ug business administration","bba regular"],
        "UGMICRO": ["microbiology department ug", "department of microbiology", "bsc microbiology", "ug microbiology","microbilogy department"],
        "PGMICRO": ["microbiology department pg", "department of microbiology", "msc microbiology", "pg microbiology"],
        "UGBIOTECH": ["biotechnology department ug", "department of biotechnology", "bsc biotechnology", "ug biotechnology","biotech","bio-tech","biotecnology department"],
        "PGBIOTECH": ["biotechnology department pg", "department of biotechnology", "msc biotechnology", "pg biotechnology"],
        "UGVISCOM": ["visual communication department ug", "department of visual communication", "bsc visual communication", "ug visual communication","viscom","visual communication department"],
        "UGCSSF": ["computer science department ug sf", "department of computer science sf", "bsc computer science sf", "ug computer science sf","computer science self"],
        "UGBCA": ["computer application department ug", "department of computer application", "bca computer applications", "ug computer applications","bca","microbilogy department"],
        "UGPHSSF": ["physical education department ug sf", "department of physical education sf", "bsc physical education sf", "ug physical education sf","phs-sf","phs sf","phs-self","physical education department self"],
        "UGENG": ["english department ug", "department of english", "ba english", "ug english","english department","eng"],
        "UGCCS": ["commerce corporate secretaryship department ug", "department of commerce corporate secretaryship", "bcom corporate secretaryship", "ug corporate secretaryship","corporate secretaryship department"],
        "PGCOM": ["commerce department pg", "department of commerce", "mcom commerce", "pg commerce","mcom"],
        "UGBBASF": ["business administration department ug sf", "department of business administration sf", "bba business administration sf", "ug business administration sf","bba sf","bba self","bba-sf"],
        "PGTAMIL": ["tamil department pg", "department of tamil", "ma tamil", "pg tamil"],
        "PGMAT": ["mathematics department pg", "department of mathematics", "msc mathematics", "pg mathematics","msc maths"],
        "PGPHY": ["physics department pg", "department of physics", "msc physics", "pg physics"],
        "PGCHE": ["chemistry department pg", "department of chemistry", "msc chemistry", "pg chemistry"],
        "PGBOT": ["botany department pg", "department of botany", "msc botany", "pg botany"],
        "PGZOO": ["zoology department pg", "department of zoology", "msc zoology", "pg zoology"],
        "PGCS": ["computer science department pg", "department of computer science", "msc computer science", "pg computer science","msc cs"],
        "PGMCA": ["computer application department pg", "department of computer application", "mca computer applications", "pg computer applications","mca"],
        "PGECO": ["economics department pg", "department of economics", "ma economics", "pg economics"],
        "UGCSR": ["computer science department ug", "department of computer science", "bsc computer science", "ug computer science","bsc cs","computer department","computer science department"],
        "UGCOMEC": ["commerce ug-ca & ec department", "department of commerce ug-ca & ec", "bcom commerce ug-ca & ec", "ug commerce ug-ca & ec"],
        "UGCPA": ["commerce professional accounting department ug sf", "department of commerce professional accounting sf", "bcom professional accounting", "ug professional accounting"]
    }
def get_response_of_department(user_input):
    for key, values in responses.items():
        for phrase in values:
            if phrase in user_input:
                return key
    return "Sorry, I don't understand department."