import streamlit as st
import genai.gemini
import genai.lama
import genai.lama_copy
import operation
import operation.dboperation
# import operation.fileoperations
import operation.fileoperations
import operation.otheroperation
import operation.preprocessing
import operation.qrsetter
import genai
import os

import operation.speech
def welcome_page():
    # st.set_page_config(page_title="Anjac_AI", layout="wide")
    data = operation.dboperation.view_student(st.session_state.user_id)
    
    #operation.dboperation.update_multifactor_status(st.session_state.user_id, st.session_state.multifactor ,secret)  # Update MFA status in the database
    # Sidebar content
    with st.sidebar:
        with st.expander(f"Welcome, {data[1]}! 🧑‍💻"):
            st.write("Choose an action:")
            with st.popover("profile"):
                st.write(f"name:{data[1]}")
                st.write(f"rollno:{st.session_state.user_id}")
            with st.popover("settings"):
                st.write()
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.page = "login"
                st.rerun()
        if "qa_list" in st.session_state and len(st.session_state.qa_list) > 3 and st.session_state.feedback == 0 and len(st.session_state.qa_list) :
            with st.popover("feedback"):
                user_id = st.text_input("Rollno",value=data[0],disabled=True)
                name = st.text_input("Your Name",value=data[1],disabled=True)
                message = st.text_area("Your Feedback")
                
                if st.button("Submit Feedback"):
                    if user_id and name and message:
                        operation.dboperation.add_feedback(user_id, name, message)
                        st.balloons()
                        st.session_state.feedback=1
                    else:
                        st.warning("Please fill all the fields to submit your feedback.")
                    st.rerun()
        st.header("Chat History")
        # if st.button("Logout"):
        #     st.session_state.authenticated = False
        #     st.session_state.page = "login"

        # Display questions and answers in reverse order
        for qa in reversed(st.session_state.qa_list):
            st.write(f"**Question:** {qa['question']}")
            st.write(f"**Answer:** {qa['answer']}")
            st.write("---")

    # Inject custom CSS for the expander
    st.markdown("""
    <style>
    .stxpander {
        position: fixed; /* Keep the expander fixed */
        top: 70px; /* Distance from the top */
        right: 10px; /* Distance from the right */
        width: 200px !important; /* Shrink the width */
        z-index: 9999; /* Bring it to the front */
    }
    .stxpander > div > div {
        background-color: #f5f5f5; /* Light grey background */
        border: 1px solid #ccc; /* Border styling */
        border-radius: 10px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    .stButton button {
        width: 90%; /* Make buttons fit nicely */
        margin: 5px auto; /* Center-align buttons */
        display: block;
        background-color: #007bff; /* Blue button */
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 14px;
        cursor: pointer;
    }
    .stpopover button {
        width: 90%; /* Make buttons fit nicely */
        margin: 5px auto; /* Center-align buttons */
        display: block;
        background-color: #007bff; /* Blue button */
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 14px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Main page user menu using expander
    

    # Main page content
    st.title("Welcome to the ANJAC AI")
    st.write(f"Hello, {data[1]}!")
    st.subheader(operation.otheroperation.get_dynamic_greeting())
    st.write("---")
    st.write(f"🎓 **Fun Fact:** {operation.otheroperation.get_fun_fact()}")
    
    # Initialize session state
    if 'qa_list' not in st.session_state:
        st.session_state.qa_list = []
    # st.header(f"{st.session_state.role} Role Content:")
    # st.text(st.session_state.role_content)
    # st.header(f"{st.session_state.role} SQL Content:")
    # st.text(st.session_state.sql_content)
    # role = st.session_state.role
    role_prompt=operation.fileoperations.read_from_file("student_role.txt")
    sql_content = operation.fileoperations.read_from_file("student_sql(2).txt")
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../files/"))

    # Define the list of files to exclude
    excluded_files = {'staff_role.txt', 'staff_sql.txt', 'student_role.txt', 'student_sql.txt', 'default.txt','staff_sql(2).txt','student_sql(2).txt'}

    # Get all files in the folder except the excluded ones
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f not in excluded_files]
    info=''
    # Print the filtered files
    for file in files:
        info += str(operation.fileoperations.read_from_file(file))
    
    chunks=operation.preprocessing.chunk_text(info)
    
    #question1 = st.text_area('Input your question:', key='input',on_change=process_and_clear)
    # submit = st.button('Ask the question')
    
    if st.button("🎤 Speak your question"):
        spoken_question = operation.speech.recognize_speech()
        # st.text(f"You said: {spoken_question}")
    else:
        spoken_question = ""

    # Text Input
    # question = st.chat_input("Ask your question") or spoken_question
    if question := st.chat_input("Ask your question") or spoken_question:
        #st.write("**ANJAC AI can make mistakes. Check important info.**")
        st.markdown("**:red[ANJAC AI can make mistakes. Check important info.]**")
    if question:
        department_id_in_user_query = operation.preprocessing.get_response_of_department(question)
        st.chat_message("user").text(question)
        print(data)
        keys = ["student_id", "name", "date of birth", "department_id","class"]
        student_dictionary = dict(zip(keys, data))
        student_info =''
        student_info += "my details "
        for key ,val in student_dictionary.items():
            student_info += f" {key} is '{val}'"
        print(student_info)
        combined_prompt = operation.preprocessing.create_combined_prompt(f" \nuser needed department_id :{department_id_in_user_query}"+student_info, sql_content)
        response = genai.lama_copy.retrive_sql_query(question,combined_prompt)

        # Display the SQL query
        # st.write("Generated SQL Query:", response)
        raw_query = response
        formatted_query = str(raw_query).replace("sql", "").strip("'''").strip()
        # print("formatted :",formatted_query)
        single_line_query = " ".join(formatted_query.split()).replace("```", "")
        # print(single_line_query)
        # Query the database
        response=str(single_line_query).replace("\n","")
        if ";" not in response:
            response = response + ";"
        print(response)
        data_sql = operation.dboperation.read_sql_query(response)
        # if len(data_sql)==0:
        #     print("done")
        #     query = str(response)
        #     trimmed_query = query.split("WHERE")[0].strip()
        #     print(trimmed_query)
        #     query=''
        #     if "timetable" in trimmed_query:
        #         query = f"SELECT * FROM timetable WHERE department_id = '{data[0][3]}'"
        #     elif "department" in trimmed_query:
        #         query = f"SELECT * FROM department"
        #     elif "student_details" in trimmed_query:
        #         query = f"SELECT * FROM student_details WHERE department_id = '{data[0][3]}'"
        #     elif "staff_details" in trimmed_query:
        #         query = f"SELECT * FROM staff_details WHERE department_id = '{data[0][3]}'"
        #     elif "student_mark_details" in trimmed_query:
        #         query = f"SELECT * FROM student_mark_details WHERE department_id = '{data[0][3]}'"
        #     print(query)
        #     data_sql = operation.dboperation.read_sql_query(query)
        if len(data_sql)==0 or 'error' in data_sql: 
            new_query=genai.lama_copy.backup_sql_query_maker("give the proper sql query without any explaination and other things ended with semicolon. "+combined_prompt,question,data_sql,response)
            print(new_query)
            raw_query = str(new_query)
            formatted_query = raw_query.replace("sql", "").strip("'''").strip()
            # print("formatted :",formatted_query)
            single_line_query = " ".join(formatted_query.split()).replace("```", "")
            print(single_line_query)

            new_query=str(single_line_query).replace("\n","")
            if ";" not in new_query:
                new_query = new_query + ";"
            # print(response)

            data_sql = operation.dboperation.read_sql_query(new_query)
        print(data_sql)
        if isinstance(data_sql, list):
            #st.write("according to,")
            #st.table(data)
            pass
            
        else:
            #st.write(data)
            # Display any errors
            pass
        # Generate response for the question and answer
        relevent_chunk=operation.preprocessing.get_relevant_chunks(question,chunks)
        relevent_chunk.append(f"{question}"+str(data_sql))
        context = "\n\n".join(relevent_chunk)
        # st.write(context)
        # st.write(question)
        from datetime import datetime
        current_datetime = datetime.now()
        # answer = genai.lama.query_lm_studio(question,f"STUDENT detail {student_info}  prompt:{role_prompt} Answer this question: {question} and the inforation is needed {context}")
        answer = genai.lama_copy.query_lm_studio(question,f"""Please interact with the user without ending the communication prematurely dont restrict the user. 
        Use the following  {student_info} use the word according to or dear statement must be in formal english. 
        current date and time  {current_datetime.strftime("%A, %B %d, %Y, at %I:%M %p").lower()} and {current_datetime.now()}.
        Format your response based on this role prompt don't provide the content inside it {role_prompt} . 
        relevent general context into your response: {context}.
        user needed department_id :{department_id_in_user_query}""")
        # answer = genai.gemini.model.generate_content(f"student name :{data[1]}  prompt:{role_prompt} Answer this question: {question} with results {str(data)}")
        result_text = answer
        #st.chat_message('ai').markdown(result_text)
        # Store the question and answer in session state
        st.session_state.qa_list.append({'question': question, 'answer': result_text})
        st.rerun()
        
    if len(st.session_state.qa_list):
        last_qa = st.session_state.qa_list[-1]  # Get the last Q&A pair
        st.chat_message("user").markdown(last_qa["question"])
        bot_response = st.chat_message("assistant").markdown(last_qa["answer"])
        
        operation.speech.speak_text(last_qa["answer"])  # Plays the answer as audio
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        selected = st.feedback("thumbs")
    else:
        st.header("How can I help you today?")

        
