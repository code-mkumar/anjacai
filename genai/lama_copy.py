import requests
import json
import streamlit as st
from datetime import datetime
import time
import operation.dboperation
import operation.fileoperations
import os
from groq import Groq

os.environ["GROQ_API_KEY"] = "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"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_with_apikey={
    "llama-3.3-70b-versatile":"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",
    "llama-3.2-3b-preview":"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"
}
# LM Studio API endpoint
LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

sql_model = {"llama-3.1-8b-instant":0, "llama-3.1-8b-instant":0}
# query_model = ["meta-llama-3.1-8b-instruct@q4_k_m", "meta-llama-3.1-8b-instruct@q4_k_m:2", "meta-llama-3.1-8b-instruct@q4_k_m:3"]
query_model = {
    "llama-3.1-8b-instant":0,
    "llama-3.1-8b-instant":0,
    "llama-3.1-8b-instant":0,
    "llama-3.1-8b-instant":0
}

# query_model = {
#     "deepseek-r1-distill-llama-8b":0,
#     "deepseek-r1-distill-llama-8b:2":0
# }
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the semantic model (use a small model for faster inference)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_cached_answer_semantically(question, threshold=0.9):
    """
    Checks for a semantically similar question in the cache.
    Returns the cached answer if a similar question is found.
    """
    conn = operation.dboperation.create_connection()
    cursor = conn.cursor()
    # Retrieve all questions and answers from the cache
    cursor.execute("SELECT id, question, answer FROM cache")
    rows = cursor.fetchall()

    if not rows:
        return None
    conn.close()
    # print(rows)
    # Encode the input question and cached questions
    question_embedding = semantic_model.encode(question, convert_to_tensor=True).to('cpu')  # Ensure it's on GPU
    cached_questions = [row[1] for row in rows]
    cached_embeddings = semantic_model.encode(cached_questions, convert_to_tensor=True).to('cpu')  # Move to GPU

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(question_embedding, cached_embeddings)

    # Check if similarities tensor is valid
    if similarities.size(0) == 0 or similarities.size(1) == 0:
        return None

    # Move to CPU for NumPy processing
    similarities_np = similarities.cpu().numpy()
    max_similarity_idx = np.argmax(similarities_np)
    max_similarity_score = similarities_np[0, max_similarity_idx]

    if max_similarity_score >= threshold:
        id = rows[max_similarity_idx][0]
        conn = operation.dboperation.create_connection()
        cursor= conn.cursor()
        cursor.execute("""
            UPDATE cache
            SET  frequency = frequency + 1, timestamp = CURRENT_TIMESTAMP
            WHERE id = ?;
        """, (id,))
        conn.commit()
        conn.close()

        return rows[max_similarity_idx][2]  # Return the cached answer

    return None

def update_cache_with_semantics(question,answer,role):
    """
    Updates the cache with semantic checking to avoid duplicate entries.
    If a similar question exists, updates its entry; otherwise, inserts a new row.
    """
    conn = operation.dboperation.create_connection()
    cursor = conn.cursor()
    similar_answer = get_cached_answer_semantically(question)
    if similar_answer:
        # If a similar question exists, update its entry
        cursor.execute("""
            UPDATE cache
            SET answer = ?, frequency = frequency + 1, timestamp = CURRENT_TIMESTAMP
            WHERE question = ?
        """, (answer, question))
        print("updated")
    else:
        # Insert a new row
        cursor.execute("""
    SELECT count(*) FROM cache;
    """)
    row_count = cursor.fetchone()[0]  # Get the actual count
    print(role)
    if not role == 'student_details' and not role == 'staff_details':
        if row_count < 100:
            cursor.execute("""
                INSERT INTO cache (question, answer, frequency)
                VALUES (?, ?, ?)
            """, (question, answer, 1))
            print("Inserted")
        else:
            # Delete entries with frequency = 1, ordered by timestamp
            cursor.execute("""
                DELETE FROM cache 
                WHERE frequency = 1
                AND timestamp = (SELECT MIN(timestamp) FROM cache WHERE frequency = 1)
            """)
            print("Deleted least recently used row")

    
    conn.commit()
    conn.close()

import random

def set_model(model=None):
    # selected_model = random.choice(list(query_model.keys()))
    selected_model = min(model,key=model.get)
    model[selected_model]+=1
    return selected_model,model

def retrive_sql_query(prompt, context):
    """
    Retrieves SQL query by interacting with the model.
    """
    global query_model
    model,models = set_model(query_model)
    query_model=models
    if not model:
        return None
    print("model:",model)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%A, %B %d, %Y, at %I:%M %p").lower()
    role = str(st.session_state.role)
    formatted_role = role.replace("_details", '')

    prompt_role = 'student counsellor' if formatted_role == 'student' else 'staff assistant'
    context_with_datetime = f"{context} Today’s date and time: {formatted_datetime}."
    client = Groq(api_key="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")  # Ensure API key is set if required
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
               {"role": "system", "content": f"You are a helpful sql query developer."},
            {"role": "user", "content": f"Context: {context_with_datetime}\n\nQuestion: {prompt}"}
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True
        )
   
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None
    try:
        query_model[model]-=1
        response=''
        for chunk in completion:  # Assuming `completion` is iterable
            try:
                # json_data = json.loads(chunk)
                content = chunk.choices[0].delta.content  # ✅ Accessing attribute correctly
                if content:
                    response += content
            except json.JSONDecodeError:
                continue
        return response
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None
def backup_sql_query_maker(context,prompt,sql_data,query):
    """
    Retrieves SQL query by interacting with the model.
    """
    global sql_model
    model,models = set_model(sql_model)
    sql_model=models
    if not model:
        return None
    print("model:",model)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%A, %B %d, %Y, at %I:%M %p").lower()
    role = str(st.session_state.role)
    formatted_role = role.replace("_details", '')

    prompt_role = 'student counsellor' if formatted_role == 'student' else 'staff assistant'
    context_with_datetime = f"{context} Today’s date and time: {formatted_datetime}."
    client = Groq(api_key="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")  # Ensure API key is set if required
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
               {"role": "system", "content": f"You are a helpful sql query developer."},
               {"role": "user", "content": f"Context: {context_with_datetime}\n\nQuestion: {prompt}\n\nworng query:{query}\n\nwrong answer:{sql_data}"}
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True
        )
   
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None
    try:
        query_model[model]-=1
        response=''
        for chunk in completion:  # Assuming `completion` is iterable
            try:
                # json_data = json.loads(chunk)
                content = chunk.choices[0].delta.content  # ✅ Accessing attribute correctly
                if content:
                    response += content
            except json.JSONDecodeError:
                continue
        return response
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def query_lm_studio(prompt, context):
    """
    Queries the LM Studio model with a word scramble game running until the first token is generated.
    """
    cached_answer = get_cached_answer_semantically(prompt)
    if cached_answer:
        with st.chat_message('ai'):
            st.markdown(cached_answer)
        return cached_answer

    global query_model
    model, models = set_model(query_model)
    query_model = models
    role = st.session_state.get("role", "user")
    prompt_role = (
        'student counsellor' if role == 'student' else 
        'staff assistant' if role == 'staff' else 
        'AI assistant'
    )

    client = Groq(api_key="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")  # Ensure API key is set if required
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a helpful {prompt_role}."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True
        )
        
        content_accumulated = ""
        with st.status("Generating data...", expanded=True) as status:
            with st.chat_message("assistant"):
                content_placeholder = st.empty()
                status.update(label="Analysing...", state="running", expanded=True)

                for chunk in completion:  # Assuming `completion` is iterable
                    try:
                        # json_data = json.loads(chunk)
                        content = chunk.choices[0].delta.content  # ✅ Accessing attribute correctly
                        if content:
                            content_accumulated += content
                            content_placeholder.markdown(content_accumulated)
                            status.update(label="Generating...", state="running", expanded=True)
                    except json.JSONDecodeError:
                        continue

            status.update(label="Generation complete!", state="complete", expanded=True)

        if context:
            update_cache_with_semantics(prompt, content_accumulated, role)
        
        query_model[model] -= 1
        return content_accumulated

    except requests.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

def query_question_maker(context,prompt):
    """
    Retrieves SQL query by interacting with the model.
    """
    global query_model
    model,models = set_model(query_model)
    query_model=models
    if not model:
        return None
    print("model:",model)
    
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are a helpful question reformer."},
            {"role": "user", "content": context + "question" + prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    try:
        response = requests.post(LM_STUDIO_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Check for HTTP errors
        response_data = response.json()
        query_model[model]-=1
        return response_data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
def word_scramble_game():
    """
    A simple word scramble game that runs until the first token is generated.
    """
    st.title("🔀 Word Scramble Game")
    st.write("Unscramble the word and type your answer!")

    words = ["streamlit", "python", "programming", "developer", "challenge"]

    if "scrambled_word" not in st.session_state:
        original_word = random.choice(words)
        scrambled_word = "".join(random.sample(original_word, len(original_word)))
        st.session_state.scrambled_word = scrambled_word
        st.session_state.original_word = original_word

    # Display scrambled word
    st.write(f"Scrambled word: **{st.session_state.scrambled_word}**")

    # Input for the user's guess
    user_guess = st.text_input("Your guess:")
    if user_guess:
        if user_guess.lower() == st.session_state.original_word:
            st.success("🎉 Correct! You unscrambled the word.")
            # Reset the game
            original_word = random.choice(words)
            scrambled_word = "".join(random.sample(original_word, len(original_word)))
            st.session_state.scrambled_word = scrambled_word
            st.session_state.original_word = original_word
        else:
            st.error("❌ Incorrect! Try again.")

    # Return True if a token is generated to exit the game loop
    return st.session_state.get("first_token_generated", False)
