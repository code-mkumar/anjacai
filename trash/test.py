import operation
import operation.dboperation

# Establish connection
conn = operation.dboperation.create_connection()
mycursor = conn.cursor()

# Define the department ID for filtering
department_id = 'UGCSR'  # Replace with the actual department ID

# # # Create a table query
# query = """
# CREATE TABLE IF NOT EXISTS cache (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     question TEXT NOT NULL,
#     answer TEXT NOT NULL,
#     frequency INTEGER NOT NULL,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# );
# """
# query ='''SELECT 
#             sd.id AS student_id,
#             sd.name AS student_name,
#             sd.class AS class,
#             s.name AS subject_name,
#             sm.quiz1,
#             sm.quiz2,
#             sm.quiz3,
#             sm.assignment1,
#             sm.assignment2,
#             sm.internal1,
#             sm.internal2,
#             sm.internal3
#         FROM 
#             student_details sd
#         JOIN 
#             student_mark_details sm ON sd.id = sm.student_id
#         JOIN 
#             subject s ON sm.subject_id = s.id
#         WHERE 
#             sd.department_id = 'PGCS' AND sd.class = 'II' AND s.id = '23PCSC412';
# '''
query='SELECT s.id, s.name, s.dob, s.department_id, s.class, d.name AS department_name FROM student_details s JOIN department_details d ON s.department_id = d.id WHERE s.student_id = '23PCS101' AND d.department_id = 'PGCS';'
# Execute the table creation query
mycursor.execute(query)

# # Define a query to fetch data (example query)
# fetch_query = "SELECT * FROM cache WHERE department_id = ?"


# # Execute the fetch query with parameter
# mycursor.execute(fetch_query, (department_id,))
result = mycursor.fetchall()

# Iterate through the results and print them
for row in result:
    print(row)

# Commit the transaction and close the connection
conn.commit()
conn.close()
# import requests
# import time

# LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# def check_model_status(model_name, timeout=10):
#     """
#     Sends a request to the LM Studio API to check if the model can accept tasks.
    
#     Parameters:
#         model_name (str): The name of the model to check.
#         timeout (int): Timeout for the request in seconds (default is 10 seconds).
    
#     Returns:
#         bool: True if the model can accept tasks (is not busy), False if busy or error.
#     """
#     payload = {
#         "model": model_name,
#         "messages": [{"role": "system", "content": "Check status"}]
#     }
#     try:
#         # Set the timeout for the request
#         response = requests.post(LM_STUDIO_API_URL, json=payload, timeout=timeout)
#         response.raise_for_status()
#         data = response.json()
#         # Assuming the API response includes a field to indicate model availability
#         return not data.get("busy", False)  # Adjust based on actual API response
#     except requests.RequestException as e:
#         print(f"An error occurred while checking {model_name}: {e}")
#         return False

# def wait_for_model_to_be_free(model_name, timeout=600, poll_interval=5, request_timeout=10):
#     """
#     Waits for the model to become free by polling the status.
    
#     Parameters:
#         model_name (str): The name of the model to check.
#         timeout (int): Maximum time to wait for the model to become free (in seconds).
#         poll_interval (int): Interval time between status checks (in seconds).
#         request_timeout (int): Timeout for each status check request (in seconds).
    
#     Returns:
#         bool: True if the model becomes free, False if it times out.
#     """
#     start_time = time.time()
#     while True:
#         if check_model_status(model_name, timeout=request_timeout):
#             print(f"Model {model_name} is free.")
#             return True
#         else:
#             elapsed_time = time.time() - start_time
#             if elapsed_time >= timeout:
#                 print(f"Timeout reached. Model {model_name} is still busy.")
#                 return False
#             print(f"Model {model_name} is busy. Waiting...")
#             time.sleep(poll_interval)  # Wait before checking again

# # Example usage
# models = ["llama-3.2-1b-instruct", "llama-3.2-1b-instruct:2", 
#           "meta-llama-3.1-8b-instruct@q4_k_m", "meta-llama-3.1-8b-instruct@q4_k_m:2", 
#           "meta-llama-3.1-8b-instruct@q4_k_m:3"]

# def is_model(models):
#     for model in models:
#         if wait_for_model_to_be_free(model):
#             return model


# import requests
# import json
# import streamlit as st
# from datetime import datetime
# import time
# import operation.dboperation


# # LM Studio API endpoint
# LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# sql_model = ["llama-3.2-1b-instruct", "llama-3.2-1b-instruct:2"]
# # query_model = ["meta-llama-3.1-8b-instruct@q4_k_m", "meta-llama-3.1-8b-instruct@q4_k_m:2", "meta-llama-3.1-8b-instruct@q4_k_m:3"]
# query_model = {
#     "meta-llama-3.1-8b-instruct@q4_k_m":0,
#     "meta-llama-3.1-8b-instruct@q4_k_m:2":0,
#     "meta-llama-3.1-8b-instruct@q4_k_m:3":0
# }

# from sentence_transformers import SentenceTransformer, util
# import numpy as np

# # Load the semantic model (use a small model for faster inference)
# semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_cached_answer_semantically(question, threshold=0.8):
#     """
#     Checks for a semantically similar question in the cache.
#     Returns the cached answer if a similar question is found.
#     """
#     conn = operation.dboperation.create_connection()
#     cursor = conn.cursor()
#     # Retrieve all questions and answers from the cache
#     cursor.execute("SELECT question, answer FROM cache")
#     rows = cursor.fetchall()

#     if not rows:
#         return None
#     conn.close()
#     # print(rows)
#     # Encode the input question and cached questions
#     question_embedding = semantic_model.encode(question, convert_to_tensor=True).to('cuda')  # Ensure it's on GPU
#     cached_questions = [row[0] for row in rows]
#     cached_embeddings = semantic_model.encode(cached_questions, convert_to_tensor=True).to('cuda')  # Move to GPU

#     # Compute cosine similarity
#     similarities = util.pytorch_cos_sim(question_embedding, cached_embeddings)

#     # Check if similarities tensor is valid
#     if similarities.size(0) == 0 or similarities.size(1) == 0:
#         return None

#     # Move to CPU for NumPy processing
#     similarities_np = similarities.cpu().numpy()
#     max_similarity_idx = np.argmax(similarities_np)
#     max_similarity_score = similarities_np[0, max_similarity_idx]

#     if max_similarity_score >= threshold:
#         return rows[max_similarity_idx][1]  # Return the cached answer

#     return None

# def update_cache_with_semantics(question,answer):
#     """
#     Updates the cache with semantic checking to avoid duplicate entries.
#     If a similar question exists, updates its entry; otherwise, inserts a new row.
#     """
#     conn = operation.dboperation.create_connection()
#     cursor = conn.cursor()
#     similar_answer = get_cached_answer_semantically(question)
#     if similar_answer:
#         # If a similar question exists, update its entry
#         cursor.execute("""
#             UPDATE cache
#             SET answer = ?, frequency = frequency + 1, timestamp = CURRENT_TIMESTAMP
#             WHERE question = ?
#         """, (answer, question))
#         print("updated")
#     else:
#         # Insert a new row
#         cursor.execute("""
#     SELECT count(*) FROM cache;
#     """)
#     row_count = cursor.fetchone()[0]  # Get the actual count

#     if row_count < 50:
#         cursor.execute("""
#             INSERT INTO cache (question, answer, frequency)
#             VALUES (?, ?, ?)
#         """, (question, answer, 1))
#         print("Inserted")
#     else:
#         # Delete entries with frequency = 1, ordered by timestamp
#         cursor.execute("""
#             DELETE FROM cache 
#             WHERE frequency = 1
#             AND timestamp = (SELECT MIN(timestamp) FROM cache WHERE frequency = 1)
#         """)
#         print("Deleted least recently used row")

    
#     conn.commit()
#     conn.close()

# import random
# def set_model(query_model):
#     selected_model = random.choice(query_model)
    
#     return selected_model

# def retrive_sql_query(prompt, context):
#     """
#     Retrieves SQL query by interacting with the model.
#     """
#     model = set_model(sql_model)
#     if not model:
#         return None

#     current_datetime = datetime.now()
#     formatted_datetime = current_datetime.strftime("%A, %B %d, %Y, at %I:%M %p").lower()
#     role = str(st.session_state.role)
#     formatted_role = role.replace("_details", '')

#     prompt_role = 'student counsellor' if formatted_role == 'student' else 'staff assistant'
#     context_with_datetime = f"{context} Today’s date and time: {formatted_datetime}."

#     headers = {"Content-Type": "application/json"}
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": f"You are a helpful {prompt_role}."},
#             {"role": "user", "content": f"Context: {context_with_datetime}\n\nQuestion: {prompt}"}
#         ],
#         "temperature": 0.7,
#         "max_tokens": 2000,
#     }

#     try:
#         response = requests.post(LM_STUDIO_API_URL, headers=headers, json=payload)
#         response.raise_for_status()  # Check for HTTP errors
#         response_data = response.json()
#         return response_data["choices"][0]["message"]["content"]
#     except requests.RequestException as e:
#         st.error(f"Request failed: {e}")
#         return None

# def query_lm_studio(prompt, context):
#     """
#     Queries the LM Studio model with streaming response.
#     """
#     cached_answer = get_cached_answer_semantically(prompt)
#     content_accumulated = ""
#     if cached_answer:
#         # st.info("Answer retrieved from cache.")
#         content_accumulated =cached_answer
#         st.chat_message('ai').markdown(content_accumulated)
#         return content_accumulated
#     # model = fetch_model(query_model)
#     # if not model:
#     #     return None
#     model = set_model(query_model)
#     role = str(st.session_state.role)
#     prompt_role = 'student counsellor' if role == 'student' else 'staff assistant'

#     headers = {"Content-Type": "application/json"}
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": f"You are a helpful {prompt_role}."},
#             {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
#         ],
#         "temperature": 0.7,
#         "max_tokens": 2000,
#         "stream": True
#     }

    

#     try:
#         with requests.post(LM_STUDIO_API_URL, headers=headers, json=payload, stream=True) as response:
#             response.raise_for_status()  # Check for HTTP errors
#             if response.status_code == 200:
#                 with st.chat_message("assistant"):
#                     content_placeholder = st.empty()
#                     for line in response.iter_lines():
#                         if line:
#                             decoded_line = line.decode("utf-8").strip()
#                             if decoded_line.startswith("data:"):
#                                 decoded_line = decoded_line[len("data:"):].strip()

#                             if decoded_line == "DONE":
#                                 return content_accumulated

#                             try:
#                                 json_data = json.loads(decoded_line)
#                                 content = json_data["choices"][0]["delta"].get("content", "")
#                                 if content:
#                                     content_accumulated += content
#                                     content_placeholder.markdown(content_accumulated)
#                             except json.JSONDecodeError:
#                                 continue
#             else:
#                 st.error(f"Error: {response.status_code} - {response.text}")
#                 return None
#         update_cache_with_semantics(prompt,content_accumulated)
#         return content_accumulated
#     except requests.RequestException as e:
#         st.error(f"An error occurred: {e}")
#         return None
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# def send_email(sender_email, sender_password, recipient_email, subject, body):
#     try:
#         # Set up the email server (Gmail SMTP server in this case)
#         smtp_server = "smtp.gmail.com"
#         smtp_port = 587

#         # Create a secure connection
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()

#         # Log in to the email account
#         server.login(sender_email, sender_password)

#         # Create the email
#         message = MIMEMultipart()
#         message["From"] = sender_email
#         message["To"] = recipient_email
#         message["Subject"] = subject

#         # Add the body to the email
#         message.attach(MIMEText(body, "plain"))

#         # Send the email
#         server.sendmail(sender_email, recipient_email, message.as_string())

#         print("Email sent successfully!")

#     except Exception as e:
#         print(f"Error sending email: {e}")
#     finally:
#         server.quit()

# # Example usage
# if __name__ == "__main__":
#     sender_email = "your_email@gmail.com"
#     sender_password = "your_email_password_or_app_password"
#     recipient_email = "recipient_email@example.com"
#     subject = "Test Email"
#     body = "This is a test email sent from Python."

#     send_email(sender_email, sender_password, recipient_email, subject, body)
