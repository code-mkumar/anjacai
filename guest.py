import streamlit as st
import genai.gemini
import genai.lama
import operation
# import operation.dboperation
# import operation.fileoperations
import operation.dboperation
import operation.fileoperations
import operation.mailoperation
import operation.otheroperation
import operation.preprocessing
import json
import genai
import operation
import os
from datetime import datetime
import random

# Function to display a dynamic welcome message

    
def guest_page():
    # Initialize session state variables
    if 'qa_list' not in st.session_state:
        st.session_state.qa_list = []
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'input' not in st.session_state:
        st.session_state.input = ""
    if 'stored_value' not in st.session_state:
        st.session_state.stored_value = ""

    # Sidebar for navigation and displaying past Q&A
    if "feedback" not in st.session_state:
        st.session_state.feedback = 0

    with st.sidebar:
        if st.button("Go to Login"):
            st.session_state.page = "login"
            st.rerun()
        
        if "qa_list" in st.session_state and len(st.session_state.qa_list) > 3 and st.session_state.feedback == 0 and len(st.session_state.qa_list) :
            # qa_list exists and its length is a multiple of 3
            
            with st.popover("feedback"):
                
                user_id = st.text_input("Email ID")
                name = st.text_input("Your Name",value=st.session_state.username,disabled=True)
                message = st.text_area("Your Feedback")
                
                if st.button("Submit Feedback"):
                    if user_id and name and message:
                        operation.dboperation.add_feedback(user_id, name, message)
                        st.balloons()
                        st.session_state.feedback=1
                        operation.mailoperation.email(name,user_id)
                    else:
                        st.warning("Please fill all the fields to submit your feedback.")
                    st.rerun()
        st.title("Chat History")
        for qa in reversed(st.session_state.qa_list):
            st.write(f"**Question:** {qa['question']}")
            st.write(f"**Answer:** {qa['answer']}")
            st.write("---")
    
            
    
    # Load text files for college and department history
    # with open("collegehistory.txt", "r") as f:
    #     collegehistory = f.read()
    # with open("departmenthistory.txt", "r") as f:
    #     departmenthistory = f.read()
    # default,default_sql=operation.fileoperations.read_default_files()
    # Display guest welcome message
    # print(feedback)
    st.title("Welcome to ANJAC AI!")
    st.write("You can explore the site as a guest, but you'll need to log in for full role-based access.")
    st.subheader(operation.otheroperation.get_dynamic_greeting())
    st.write("---")
    st.write(f"🎓 **Fun Fact:** {operation.otheroperation.get_fun_fact()}")
    # Ask for the user's name
    if not st.session_state.username:
        st.session_state.username =''
    name = ''
    if not name and not st.session_state.username:
        name=st.text_input('Enter your name:', placeholder='John', key='name')
        st.session_state.username = name
        if name :
            st.rerun()
       
    
        # st.write(f"Hello, {name}!")

    # Process questions if the username is set
    if st.session_state.username:
        role_prompt = operation.fileoperations.read_from_file('default.txt')
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../files/"))

        # Define the list of files to exclude
        excluded_files = {'staff_role.txt', 'staff_sql.txt', 'student_role.txt', 'student_sql.txt', 'default.txt','sy.txt','staff_sql(2).txt','student_sql(2).txt'}

        # Get all files in the folder except the excluded ones
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f not in excluded_files]
        info=''
        # Print the filtered files
        for file in files:
            info += str(operation.fileoperations.read_from_file(file))
        
        # chunks=operation.preprocessing.chunk_text(str(info))
        chunks = operation.preprocessing.chunk_text_by_special_character(str(info))
        #st.write(chunks)
        # st.write(chunks)
        st.write(f"Hello, {st.session_state.username}!")
        # chunks = operation.preprocessing.chunk_text(f"{collegehistory}\n{departmenthistory}")
        # question = st.chat_input("Ask your question")
       
        if question := st.chat_input("Ask your question"):
            #st.write("**ANJAC AI can make mistakes. Check important info.**")
            st.markdown("**:red[ANJAC AI can make mistakes. Check important info.]**")
        
        if question:
            # Retrieve relevant chunks
            st.chat_message('user').markdown(question)
            k = 3  
            keywords = ['more', 'most', 'details', 'full information']
            if any(keyword in question.lower() for keyword in keywords):
                k = 3 
            previous = ",".join(qa_item["question"] for qa_item in st.session_state.qa_list)
            # Get relevant chunks based on the question
            relevant_chunks = operation.preprocessing.get_relevant_chunks(question, chunks, k, previous)

            # Display the relevant chunks and their size
            #st.write(relevant_chunks)
            rel_chunk_size = len("\n\n".join(relevant_chunks))  # Size in characters
            #st.write("Size:", rel_chunk_size)
            n=500
            # While the size exceeds the threshold, keep splitting and processing
            while rel_chunk_size > 2000:
                # Split the current relevant chunks into smaller parts
                rechunk = operation.preprocessing.chunk_text("\n\n".join(relevant_chunks),n)
                st.info("Rechunked text:")
                st.info(rechunk)
                
                # Get the new relevant chunks after rechunking
                relevant_chunks = operation.preprocessing.get_relevant_chunks_re(question, rechunk)
                
                # Display the length of the new relevant chunks
                #st.write("New relevant chunk size:", len(" ".join(relevant_chunks)))
                st.info("New relevant chunks:")
                st.info(" ".join(relevant_chunks))
                
                # Update the chunk size for the next iteration
                rel_chunk_size = len("\n\n".join(relevant_chunks))  # Size in characters
                n-=100

            if relevant_chunks :
                relevant_chunks.append("college name:'AYYA NADAR JANAKI AMMAL COLLEGE' college loaction:'srivilliputur road ,sivakasi'")
            # relevant_chunks.append()

            context = "\n\n".join(relevant_chunks)

            # Display relevant context
            # st.write("Relevant context:")
            # st.write(context)

            # Query LM Studio for the answer
            
                
            result_text = genai.lama.query_lm_studio(question,context)          
            # Store the question and answer in session state
            st.session_state.qa_list.append({'question': question, 'answer': result_text})
            # st.rerun()
        
        if len(st.session_state.qa_list):
            last_qa = st.session_state.qa_list[-1]  # Get the last Q&A pair
            st.chat_message("user").markdown(last_qa["question"])
            bot_response = st.chat_message("assistant").markdown(last_qa["answer"])
            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
            selected = st.feedback("thumbs")
        else:
            st.header("How can I help you today?")


                
