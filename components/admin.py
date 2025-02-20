import streamlit as st
import os
import sqlite3
import operation
import operation.dboperation
import operation.fileoperations
#import operation.dboperation
#import operation.fileoperations
from datetime import date

import operation.preprocessing
import operation.qrsetter
def admin_page():
    # st.set_page_config(page_title="Admin Dashboard", layout="wide")
    #secret, role, name = None#operation.dboperation.get_user_details(st.session_state.user_id)
    # operation.dboperation.update_multifactor_status(st.session_state.user_id, st.session_state.multifactor ,secret)  # Update MFA status in the database
    # Sidebar content
    data = operation.dboperation.view_admins()
    with st.sidebar:
        with st.expander(f"Welcome, {data[0][0]}! 🧑‍💻"):
            st.write("Choose an action:")
            with st.popover("profile"):
                st.write(f"INITIAL :{st.session_state.user_id}")
                st.write(f"name :{data[0][0]}")
                
            with st.popover("settings"):
                st.write("update user data")
                if data[0][2] == True and data[0][2]:
                    otp = st.text_input("enter the otp" ,type='password')
                    if operation.qrsetter.verify_otp(data[0][3],otp):
                        value = st.toggle("Disable MFA", value=data[0][2], on_change=lambda: operation.dboperation.mfa_update(data[0][0], "admin_details", 0))
                        password = st.text_input("enter the new password",type="password")
                        operation.dboperation.change_pass(password,st.session_state.user_id)
                        st.success("changed successfully!!!")
                    else:
                        st.error("enter the correct otp...")
                else:
                    password = st.text_input("enter the password" ,type='password')
                    if password == data[0][1]:
                        value = st.toggle("enable MFA", value=data[0][2], on_change=lambda: operation.dboperation.mfa_update(data[0][0], "admin_details", 1))
                        password = st.text_input("enter the new password",type="password")
                        if not password == '' :
                            operation.dboperation.change_pass(password,st.session_state.user_id)
                            st.success("changed successfully!!!")
                    else:
                        st.error("enter the correct otp...")
        st.header("Admin Modules")
        module = st.radio(
            "Select Module",
            options=["File Upload and Edit", "Database Setup", "Query Area","admin data", "Logout"]
        )
       

    # Main page content
    st.title("Admin Dashboard")
    st.write(f"Welcome, admin! 👋")

    
# Ensure required files exist

    # # Ensure required files exist
    # for file_name in ["collegehistory.txt", "departmenthistory.txt", "syllabus.txt"]:
    #     if not os.path.exists(file_name):
    #         with open(file_name, "w") as f:
    #             f.write("")  # Create an empty file

    # Proceed with other operations
    
# List of text files
                           
    if module=="File Upload and Edit":
        st.write("Current working directory:", os.getcwd())
        st.subheader("File Creation")
        f1=st.text_input("Enter the file name")
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../files/{f1}.txt"))
        # departments = operation.dboperation.view_departments()
        # department_names=[department[1] for department in departments]
        # formatted_text_list=[]
        # for dept in departments:
        #         formatted_text = f"""
        #         ---
        #         **Department Name:** {dept[1]}
        #         **Type:** {dept[2]}
        #         **phone no:** {dept[3]}"""
        #         formatted_text_list.append(formatted_text)
        
        # for department_name,content in zip(department_names,formatted_text_list):
        #     operation.fileoperations.write_to_file(f"{department_name}.txt",content)
        if f1:
            operation.fileoperations.write_to_file(f"{f1}.txt")

        st.subheader("File Upload and Edit Module")
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../files/"))
        
# Get all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        #print("Files in folder:", files)
         # Selection of category to save the file
        category = st.selectbox(
        "Select the category to save the uploaded file:",
        # ["collegehistory.txt", "departmenthistory.txt", "syllabus.txt"]
        files
    )
        
    # File uploader
        uploaded_file = st.file_uploader(
        "Upload a PDF, Word, or Text file", type=["pdf", "docx", "txt"]
    )

        if uploaded_file:
        # Read and display the content of the uploaded file
            # if uploaded_file.type == "application/pdf":
            #     import PyPDF2
            #     pdf_reader = PyPDF2.PdfReader(uploaded_file)
            #     file_content = "".join([page.extract_text() for page in pdf_reader.pages])
            # elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            #     from docx import Document
            #     doc = Document(uploaded_file)
            #     file_content = "\n".join([p.text for p in doc.paragraphs])
            # else:
            #     file_content = uploaded_file.read().decode('utf-8')
            file_content = operation.fileoperations.file_to_text(uploaded_file)

            st.text_area("Uploaded File Content", value=file_content, height=300, disabled=True)

        # Section for editing file content
            edited_content = st.text_area("Edit File Content", value=file_content, height=300)

            if st.button("Save File"):
                file_path = os.path.join(folder_path, category)
                st.write(file_path)
                with open(file_path, "a", encoding='utf-8') as f:
                    f.write(edited_content)
                st.success(f"File content saved to {category} successfully!")

    # Section for managing existing files
        st.subheader("Manage Existing Files")
        existing_file = st.selectbox(
        "Select a file to view or edit:",
        # ["collegehistory.txt", "departmenthistory.txt", "syllabus.txt"]
        files
    )

        # Open File Button
        
        if st.button("Open File" ,key="open"):
            file_path = os.path.join(folder_path, existing_file)
            with open(file_path, "r", encoding='utf-8') as f:
                existing_content = f.read()
            edited_existing_content = st.text_area("Edit Existing File Content", value=existing_content, height=300,disabled=True,key="read")
            # if st.button("update Content",key="update1"):
            #     file_path = os.path.join(folder_path, existing_file)
            #     with open(file_path, "w", encoding='utf-8') as f:
            #         f.write(edited_existing_content)
            #     st.success(f"Content of {existing_file} updated successfully!")
        if st.checkbox("update"):
            file_path = os.path.join(folder_path, existing_file)
            with open(file_path, "r", encoding='utf-8') as f:
                existing_content = f.read()
            edited_existing_content = st.text_area("Edit Existing File Content", value=existing_content, height=300,key="update")
            if st.button("update Content",key="update2"):
                file_path = os.path.join(folder_path, existing_file)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(edited_existing_content)
                st.success(f"Content of {existing_file} updated successfully!")
                
            # if st.button("Update File"):
            #     try:
            #         with open(file_path, "w", encoding='utf-8') as f:
            #             f.write(edited_existing_content)
            #         st.success(f"Content of {existing_file} updated successfully!")
            #     except Exception as e:
            #         st.write(e)
            #         print("***********************************************",e)
    # Deletion section
        st.subheader("Delete File Content")
        file_to_delete = st.selectbox(
        "Select a file to delete content:",
        # ["collegehistory.txt", "departmenthistory.txt", "syllabus.txt"]
        files
    )

        if st.button("Delete Content"):
            file_path = os.path.join(folder_path, file_to_delete)
            with open(file_path, "w", encoding='utf-8') as f:
                f.write("")
            st.success(f"Content of {file_to_delete} deleted successfully!")

        import openpyxl
        

        # Define folder path
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../files/"))

        # Get all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        st.subheader("Add Data to the Department")

        # Upload Excel File
        excel = st.file_uploader("Upload the Excel")

        if excel:
            wb = openpyxl.load_workbook(excel)
            sheet = wb.active  # Assume data is in the first sheet

            # Extract column headers
            headers = [cell.value for cell in sheet[1]]

            # Extract department entries
            departments = []
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row[1]:  # Ensure department name exists
                    departments.append(row)

            # Generate structured paragraph for each department
            formatted_departments = {}

            for dept in departments:
                formatted_text = f"""
                **Department Name:** {dept[1]}
                **Type:** {dept[2]}
                **Level:** {dept[3]}

                **Present HOD:** {dept[4]}
                
                **Previous HODs and Experience:** {dept[5]}
                
                **Faculty List:** {dept[6]}
                
                **Achievements:** {dept[7]}
                
                **Notable Alumni:** {dept[8]}
                
                **Course Fees:** {dept[9]}
                
                **Mission:** {dept[10]}
                
                **Vision:** {dept[11]}
                
                **Objectives:** {dept[12]}
                
                **Eligibility for Admission:** {dept[13]}
                """
                formatted_departments[dept[1]] = formatted_text.strip()

            # Select department file and write data
            if "dept_index" not in st.session_state:
                st.session_state.dept_index = 0

            # Show the current department details
            department_names = list(formatted_departments.keys())

            if st.session_state.dept_index < len(department_names):
                current_dept = department_names[st.session_state.dept_index]
                st.write(f"### Department: {current_dept}")

                # Select file corresponding to the department
                department_file = st.selectbox(f"Select file for {current_dept}", files, key=f"d_{current_dept}")

                # Read existing content from the selected file
                file_path = os.path.join(folder_path, department_file)
                existing_content = ""
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        existing_content = f.read()

                # Display existing content in a disabled text area
                st.text_area("Existing Data", existing_content, height=150, disabled=True)

                # Display new data in a text area
                new_data = st.text_area("New Data to Add", formatted_departments[current_dept], height=250)

                col1, col2 = st.columns(2)

                # Save button
                with col1:
                    if st.button("Save & Next"):
                        with open(file_path, "a") as f:
                            f.write("\n\n" + new_data)
                        st.success(f"✅ Data saved to {department_file} successfully!")
                        
                        # Move to the next department
                        st.session_state.dept_index += 1
                        st.rerun()

                # Cancel button
                with col2:
                    if st.button("Cancel"):
                        st.session_state.dept_index = st.session_state.dept_index+1  # Exit process
                        st.warning("🚫 Process Cancelled!")
                        st.rerun()

            else:
                st.success("✅ All departments have been saved!")
   
     

        st.title("🔖 Text Chunking App with Special Character Separator")
        st.write("Upload a text file and choose a separator to split the content into chunks.")

        # Folder path
        import re
        

        # Folder Path
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../files/"))

        # Excluded Files
        excluded_files = {
            'staff_role.txt', 'staff_sql.txt', 'student_role.txt', 'student_sql.txt',
            'default.txt', 'syllabus.txt', 'staff_sql(2).txt', 'student_sql(2).txt', 'data.txt'
        }

        # Read All Valid Files into a Dictionary
        data = {}
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file not in excluded_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data[file] = content

        # Define Separator
        separator = "---"

        # Chunk the Text Data
        chunked = {}
        for file, content in data.items():
            chunked[file] = re.split(rf'\s*{re.escape(separator)}\s*', content)

        # Store Updated Chunks
        updated_chunks = {}

        for file, chunks in chunked.items():
            modified_chunks = []
            for i, chunk in enumerate(chunks):
                with st.expander(f"{file} - Chunk {i}"):
                    edited_chunk = st.text_area("Edit", value=chunk, key=f"{file}-{i}")
                    modified_chunks.append(edited_chunk)  # Store modified chunk
            
            updated_chunks[file] = modified_chunks  # Store all modified chunks per file

        # Save Changes When User Clicks Button
        if st.button("Save Changes"):
            for file, chunks in updated_chunks.items():
                result_string = f"\n{separator}\n".join(chunks)
                file_path = os.path.join(folder_path, file)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(result_string)

                st.success(f"✅ {file} updated successfully!")
            st.rerun()




    elif module == "Database Setup":
        

       
        
        # Streamlit app
        st.title("Department and Related Tables Management")
        
        # Create tables if they don't exist
        # operation.dboperation.create_main_tables()
        if st.button('create table'):
            operation.dboperation.create_tables()
        
        

        
        # Add a Department
        with st.expander("Add a New Department"):
            department_id = st.text_input("Department Id:")
            name = st.text_input("Department Name:")
            graduate_level = st.selectbox("Graduate Level:", ["UG", "PG"])
            phone = st.text_input("Phone Number:")
        
            if st.button("Add Department"):
                if department_id and name and graduate_level and phone:
                    operation.dboperation.add_department(department_id,name,graduate_level,phone)
                    st.success(f"Department '{name}' added successfully!")
                else:
                    st.error("Please fill all the fields.")
        
        # Display existing departments
        departments = operation.dboperation.view_departments()
        if departments:
            with st.expander("Existing Departments"):
                for dept in departments:
                    st.write(f"ID: {dept[0]}, Name: {dept[1]}, Graduate Level: {dept[2]}, Phone: {dept[3]}")
        
            # Select a department ID for further actions
            selected_department_id = st.selectbox(
                "Select Department ID for Adding Staff, Timetable, or Subjects:", 
                [dept[0] for dept in departments]
            )
            selected_dept = next((dept for dept in departments if dept[0] == selected_department_id), None)
            graduate_level = selected_dept[2]
            file_path = st.file_uploader("upload")
            lines=''
            if file_path:          
                lines = [line.strip() for line in file_path.readlines() if line.strip()]
            # Add Staff to the selected department
            with st.expander("Add Staff to Selected Department"):
                staff_id = st.text_input("Staff Id:")
                selected_data = st.selectbox("select the line",lines)
                staff_name = st.text_input("Staff Name:",value=selected_data)
                designation = st.text_input("Designation:")
                staff_phone = st.text_input("Phone:")
                
                if st.button("Add Staff"):
                    
                    if staff_id and staff_name and designation and staff_phone:
                        operation.dboperation.add_staff(staff_id,staff_name,designation,selected_department_id,"pass_staff")
                        st.success(f"Staff '{staff_name}' added to Department ID {selected_department_id}!")
                    else:
                        st.error("Please fill all the fields.")
        
            # Add Timetable to the selected department
            with st.expander("Add Timetable to Selected Department"):
                # timetable_id = st.text_input("Time Table Id:")
                day = st.selectbox("day:",["monday","tuesday","wednesday","thursday","friday","saturday"])
                if day =="saturday":
                    time = st.selectbox("Time:",["9.00-9.45","9.45-10.30","10.30-11.15","11.20-12.10","12.10-1.00"])
                else:
                    time = st.selectbox("Time:",["10-11","11-12","12-1","2-3","3-4","4-5"])
                subject = st.text_input("Subject:")
        
                if graduate_level == "PG": 
                    class_name = st.selectbox("Class:", ["I", "II"],key="PG_timetable") 
                else: 
                    class_name = st.selectbox("Class:", ["I", "II", "III"],key="UG_timetable")
                new_staff = st.selectbox("select the staff",[i[0] for i in operation.dboperation.view_staffs(department_id)])
                if st.button("Add Timetable"):
                    if  day and time and subject:
                        operation.dboperation.add_timetable(day, time, subject,class_name, selected_department_id,staff_id=new_staff)
                        st.success(f"Timetable for '{day} at {time}' added to Department ID {selected_department_id}!")
                    else:
                        st.error("Please fill all the fields.")
        
            # Add Subject to the selected department
            with st.expander("Add Subject to Selected Department"):
               
                subject_name = st.text_input("Subject Name:")
                subject_code = st.text_input("Subject Code:")
                class_name=''
                if graduate_level == "PG": 
                    class_name = st.selectbox("Class:", ["I", "II"],key="PG_subject") 
                else: 
                    class_name = st.selectbox("Class:", ["I", "II", "III"],key="UG_subject")
                if st.button("Add Subject"):
                    if subject_name and subject_code:
                        operation.dboperation.add_subject( subject_id=subject_code, department_id=selected_department_id,name=subject_name,class_name=class_name)
                        st.success(f"Subject '{subject_name}' added to Department ID {selected_department_id}!")
                    else:
                        st.error("Please fill all the fields.")

            # Add Student to the selected department
            with st.expander("Add Student to Selected Department"):
                # Define the range of dates
                min_date = date(2000, 1, 1)  # Minimum date
                max_date = date(2050, 12, 31)  # Maximum date
                rollno = st.text_input("Student Rollno:")
                name = st.text_input("Student Name:")
                dob=st.date_input("Date of Birth",min_value=min_date,max_value=max_date)
                if graduate_level == "PG": 
                    class_name = st.selectbox("Class:", ["I", "II"],key="PG_student") 
                else: 
                    class_name = st.selectbox("Class:", ["I", "II", "III"],key="UG_student")
                
                if st.button("Add Student"):
                    if rollno and name and dob and class_name:
                        operation.dboperation.add_student( rollno, name,dob,selected_department_id,class_name)
                        st.success(f"Student '{name}' added to Department ID {selected_department_id}!")
                    else:
                        st.error("Please fill all the fields.")

            # Add marks to the selected department
            with st.expander("Add Marks to Selected Student"):
                st.subheader("Mark Entry")
                if graduate_level == "PG": 
                    class_name = st.selectbox("Class:", ["I", "II"],key="PG_mark") 
                else: 
                    class_name = st.selectbox("Class:", ["I", "II", "III"],key="UG_mark")
                
                if class_name:
                    cycle = st.selectbox("Cycle", ["1", "2", "3"])
                    
                    if cycle:
                        # Fetch students for the selected department and class
                        students = operation.dboperation.view_students(selected_department_id, class_name)
                        students_id = [i[0] for i in students]  # List of student IDs
                        students_names = [i[1] for i in students]  # List of student names
                        subjects = operation.dboperation.view_subjects(selected_department_id,class_name)
                        subjects_id = [i[0] for i in subjects]  # List of student IDs
                        # Select student by name
                        selected_student_id = st.selectbox("Select Student", students_id)
                        subject_id = st.selectbox("select subject",subjects_id)
                        if selected_student_id and subject_id:
                            # Display the roll number for the selected student
                            id = st.text_input("Roll No:", value=selected_student_id, disabled=True)

                            # Inputs for marks

                            quiz = st.number_input("Quiz Marks", min_value=0.0, max_value=5.0, step=1.0)
                            if not cycle=="3":
                                assignment = st.number_input("Assignment Marks", min_value=0.0, max_value=10.0, step=1.0)
                            internal_marks = st.number_input("Internal Marks", min_value=0.0,max_value=30.0,step=1.0)

                            # Check if all inputs are filled before submitting
                            if st.button("submit"):
                                if id and quiz and assignment and internal_marks and cycle:
                                    # Add marks to the database
                                    operation.dboperation.add_marks(id,subject_id, cycle, quiz, assignment, internal_marks)
                                    st.success("Marks added successfully!")

    elif module == "Query Area":
        import pandas as pd

        st.title("View Section")

        if st.checkbox("view all departments"):
            departments = operation.dboperation.view_departments()
            st.table(departments)
        
        st.title("Dynamic Department Viewer")
        
        # Fetch department details
        # Fetch department details
        departments = operation.dboperation.view_departments()
        # print(departments)
        # Assuming departments contain tuples like (department_id, department_name, graduate_level, phone)
        department_dict = {row[1]: (row[0], row[2], row[3]) for row in departments}  # Mapping name to (ID, graduate_level, phone)

        # Select department
        department_name = st.selectbox("Select Department", list(department_dict.keys()))
        department_id, graduate_level, phone = department_dict[department_name]

        # Display and edit department details
        st.subheader("Department Details")
        if st.button("View Department Details"):
            st.write(f"Details for Department: {department_name} (ID: {department_id})")
            st.write(f"Department ID: {department_id}, Name: {department_name}")
            st.write(f"Graduate Level: {graduate_level}")
            st.write(f"Phone: {phone}")
        department_ids= [row[0] for row in departments]
        # Edit department details
        if st.checkbox("Edit Department Details"):
            department_dict = {row[0]: (row[1], row[2], row[3]) for row in departments}
            id = st.selectbox("select the department",department_dict.keys())
            new_name = st.text_input("New Department Name", value=department_dict[id][0])
            new_graduate_level = st.selectbox("New Graduate Level", ["UG", "PG", "Research"], index=["UG", "PG", "Research"].index(graduate_level))
            new_phone = st.text_input("New Department Phone", value=phone)
            
            if st.button("Update Department"):
                operation.dboperation.update_department(id, new_name, new_graduate_level, new_phone)
                st.success("Department updated successfully.")

        # Delete department
        if st.checkbox("Delete Department"):
            if st.button("Delete Department"):
                operation.dboperation.delete_department(department_id)
                st.success("Department deleted successfully.")

        
        # Fetch and edit staff details
        st.subheader("Staff Details")
        staff_data = operation.dboperation.view_staffs(department_id)
        st.table(staff_data)
        
        staff_ids = [record[0] for record in staff_data]  # Assuming `record[0]` is the `staff_id`
        selected_staff_id = st.selectbox("Select Staff ID to Update:", options=staff_ids)
        if st.checkbox("update staff"):
            # Pre-fill fields based on selected staff
            selected_staff = next((record for record in staff_data if record[0] == selected_staff_id), None)
            if selected_staff:
                with st.form("update_staff_form"):
                    st.write("Update Staff Details")
                    
                    staff_id = st.text_input("Staff ID (required):", value=selected_staff[0], disabled=True)
                    name = st.text_input("Name:", value=selected_staff[1])  # Assuming `record[1]` is `name`
                    designation = st.text_input("Designation:", value=selected_staff[2])  # Adjust indices as needed
                    department_id = st.text_input("Department ID:", value=selected_staff[3])
                    password = st.text_input("Password:", type="password")
                    mfa = st.text_input("MFA:", value=selected_staff[5])  # Adjust index if `mfa` exists
                    secd = st.text_input("Secondary Details:", value=selected_staff[6])
                    phone_no = st.text_input("Phone Number:", value=selected_staff[7])
                    email = st.text_input("Email:", value=selected_staff[8])

                    submitted = st.form_submit_button("Update Staff")
                    if submitted:
                        if staff_id:
                            operation.dboperation.update_staff(
                                staff_id,
                                name ,
                                designation ,
                                department_id ,
                                password ,
                                mfa ,
                                secd ,
                                phone_no ,
                                email 
                            )
                            st.success(f"Staff {staff_id} details updated.")
                        else:
                            st.error("Staff ID is required for updating details.")
        if st.checkbox("Delete staff"):
            st.subheader("Delete Staff")
            if st.button("Delete Staff"):
                if selected_staff_id:
                    operation.dboperation.delete_staff(selected_staff_id)
                    st.success(f"Staff {selected_staff_id} deleted.")
                else:
                    st.error("Staff ID is required to delete a staff member.")
        
        # Tab for deleting staff
            

            # Timetable View
        st.subheader("Timetable Details")

        # Select class based on graduate level
        class_name = ''
        if graduate_level == "PG":
            class_name = st.selectbox("Class:", ["I", "II"],key="PG_time")
        else:
            class_name = st.selectbox("Class:", ["I", "II", "III"],key="UG_time")

        if class_name:
            # print(class_name)
            timetable_data = operation.dboperation.view_timetable(department_id, class_name)
            # st.table(timetable_data)
            # print(timetable_data)
            df = pd.DataFrame(timetable_data, columns=['ID', 'Day', 'Time', 'Subject', 'Section', 'Department','Staff'])

            df_weekdays = df[df['Day'].isin(['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])]

            # Concatenate subjects for duplicate time slots
            df_agg = df_weekdays.groupby(['Day', 'Time'],observed=False)['Subject'].apply(lambda x: ', '.join(x)).reset_index()

            # Create a pivot table
            pivot_table = df_agg.pivot(index='Day', columns='Time', values='Subject')

            # Sort the index and columns to ensure correct order
            pivot_table = pivot_table.reindex(index=['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])
            pivot_table = pivot_table.sort_index(axis=1)

            df_saturday = df[df['Day'] == 'saturday']

            # Define the custom order for time slots
            time_order = ['9.00-9.45', '9.45-10.30', '10.30-11.15', '11.20-12.10', '12.10-1.00']

            # Ensure the time column follows the correct order
            df_saturday['Time'] = pd.Categorical(df_saturday['Time'], categories=time_order, ordered=True)

            # Sort the DataFrame by the custom time order
            df_saturday_sorted = df_saturday.sort_values('Time')

            # Concatenate subjects for duplicate time slots
            df_saturday_agg = df_saturday_sorted.groupby(['Day', 'Time'],observed=False)['Subject'].apply(lambda x: ', '.join(x)).reset_index()

            # Create a pivot table for Saturday
            saturday_pivot = df_saturday_agg.pivot(index='Day', columns='Time', values='Subject')

            st.subheader("weekdays time table")
            st.table(pivot_table)
            st.subheader("staurday order")
            st.table(saturday_pivot)
            
                # Update timetable
            st.subheader("Manage Timetable")
            if st.checkbox("Update Timetable"):
                st.write("Update Timetable Entry")
                entry_to_update = st.selectbox(
                    "Select Entry to Update",
                    df.to_dict('records'),
                    format_func=lambda x: f"{x['ID']} {x['Day']} {x['Time']} - {x['Subject']} {x['Staff']}"
                )
                if entry_to_update:
                    with st.form("update_timetable_form"):
                        timetable_id = entry_to_update["ID"]
                        new_day = st.selectbox("Day", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"], index=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"].index(entry_to_update['Day']))
                        new_time=''
                        if new_day == "saturday":
                            new_time = st.selectbox("Time",['9.00-9.45','9.45-10.30','10.30-11.15', '11.20-12.10','12.10-1.00'], index=['9.00-9.45','9.45-10.30','10.30-11.15', '11.20-12.10','12.10-1.00'].index(entry_to_update["Time"]))
                        else:
                            new_time = st.selectbox("Time",['10-11','11-12','12-1','2-3','3-4'], index=['10-11','11-12','12-1','2-3','3-4'].index(entry_to_update["Time"]))
                        new_subject = st.text_input("Subject", value=entry_to_update["Subject"])
                        new_staff = st.selectbox("select the staff",[i[0] for i in operation.dboperation.view_staffs(department_id)])
                        submitted = st.form_submit_button("Update")
                        if submitted:
                            operation.dboperation.update_timetable(staff_id=new_staff,timetable_id=timetable_id,department_id=department_id, class_name=class_name,day= new_day,time= new_time, subject=new_subject)
                            st.success("Timetable entry updated successfully.")

            # Delete timetable
            if st.checkbox("Delete Timetable"):
                st.write("Delete Timetable Entry")
                entry_to_delete = st.selectbox(
                    "Select Entry to Delete",
                    df.to_dict('records'),
                    format_func=lambda x: f"{x['ID']}{x['Day']} {x['Time']} - {x['Subject']}"
                )
                if entry_to_delete:
                    if st.button("Delete"):
                        operation.dboperation.delete_timetable(entry_to_delete["ID"])
                        st.success("Timetable entry deleted successfully.")
        else:
            st.warning("No timetable found for this department.")


        
        # Subject Details
        st.subheader("Subject Details")

        # View Subjects
        if st.button("View Subject Details"):
            subjects = operation.dboperation.view_subjects_department(department_id)
            if subjects:
                st.write(f"Subject Details for Department: {department_name}")
                subject_df = pd.DataFrame(subjects, columns=["Subject ID", "Name","Department ID", "Class"])
                st.dataframe(subject_df)
            else:
                st.warning(f"No subjects found for Department: {department_name}")

        # Update Subject
        if st.checkbox("Update Subject"):
            subjects = operation.dboperation.view_subjects_department(department_id)
            if subjects:
                subject_ids = [subject[0] for subject in subjects] 
                print(subject_ids) # Assuming Subject ID is the first field
                selected_subject_id = st.selectbox("Select Subject ID", subject_ids)
                
                # Pre-fill current details
                selected_subject = next((subject for subject in subjects if subject[0] == selected_subject_id), None)
                if selected_subject:
                    current_department = selected_subject[1]  # Assuming Subject Name is the second field
                    current_name = selected_subject[2]  # Assuming Class is the third field
                    current_class = selected_subject[3]
                    new_name = st.text_input("Enter New Subject Name", value=current_name)
                    if graduate_level == "PG":
                        class_name = st.selectbox("Class:", ["I", "II"], index=["I", "II"].index(current_class),key="PG_sub")
                    else:
                        class_name = st.selectbox("Class:", ["I", "II", "III"], index=["I", "II", "III"].index(current_class),key="UG_sub")
                    
                    if st.button("Update Subject"):
                        operation.dboperation.update_subject(selected_subject_id, department_id, new_name, class_name)
                        st.success("Subject details updated successfully.")
            else:
                st.warning("No subjects available for update.")

        #
        # Delete Subject
        if st.checkbox("Delete Subject"):
            subjects = operation.dboperation.view_subjects_departemnt(department_id)
            if subjects:
                subject_ids = [subject[0] for subject in subjects]  # Assuming Subject ID is the first field
                selected_subject_id = st.selectbox("Select Subject ID to Delete", subject_ids)
                
                if st.button("Delete Subject"):
                    operation.dboperation.delete_subject(selected_subject_id)
                    st.success("Subject deleted successfully.")
            else:
                st.warning("No subjects available for deletion.")
        # Student Details
        st.subheader("Student Details")

        # Fetch Students Based on Department and Class
        if graduate_level == "PG":
            class_name = st.selectbox("Select Class:", ["I", "II"],key="PG_stu")
        else:
            class_name = st.selectbox("Select Class:", ["I", "II", "III"],key="UG_stu")

        if class_name:
            students = operation.dboperation.view_students(department_id, class_name)
            if students:
                st.write(f"Students in {department_name}, Class {class_name}:")
                student_df = pd.DataFrame(students, columns=["Student ID", "Name", "Date of Birth", "Department ID", "Class"])
                st.dataframe(student_df)
            else:
                st.warning(f"No students found for {department_name}, Class {class_name}.")

        # Update Student
        if st.checkbox("Update Student"):
            if students:
                student_ids = [student[0] for student in students]  # Assuming Student ID is the first field
                selected_student_id = st.selectbox("Select Student ID to Update:", student_ids)

                # Pre-fill fields for the selected student
                selected_student = next((student for student in students if student[0] == selected_student_id), None)
                if selected_student:
                    current_name = selected_student[1]  # Assuming Name is the second field
                    current_dob = selected_student[2]  # Assuming Date of Birth is the third field
                    current_class = selected_student[4]  # Assuming Class is the fifth field

                    with st.form("update_student_form"):
                        st.write("Update Student Details")
                        student_id = st.text_input("Student ID (required):", value=selected_student_id, disabled=True)
                        name = st.text_input("Name:", value=current_name)
                        dob = st.date_input("Date of Birth:", value=pd.to_datetime(current_dob))
                        class_name=''
                        if graduate_level == "PG":
                            class_name = st.selectbox("Class:", ["I", "II"], index=["I", "II"].index(current_class),key="PG_stu_update")
                        else:
                            class_name = st.selectbox("Class:", ["I", "II", "III"], index=["I", "II", "III"].index(current_class),key="UG_stu_update")

                        submitted = st.form_submit_button("Update Student")
                        if submitted:
                            if student_id:
                                operation.dboperation.update_student(student_id, name, dob, department_id, class_name)
                                st.success(f"Student {student_id} details updated successfully.")
                            else:
                                st.error("Student ID is required to update details.")
            else:
                st.warning("No students available for update.")

        # Delete Student
        if st.checkbox("Delete Student"):
            if students:
                student_ids = [student[0] for student in students]  # Assuming Student ID is the first field
                selected_student_id = st.selectbox("Select Student ID to Delete:", student_ids)

                if st.button("Delete Student"):
                    operation.dboperation.delete_student(selected_student_id)
                    st.success(f"Student {selected_student_id} deleted successfully.")
            else:
                st.warning("No students available for deletion.")

# Subject-Wise Marks Management
        st.subheader("Subject-Wise Marks Management")

        # Select Department and Class
        if graduate_level == "PG":
            class_name = st.selectbox("Select Class:", ["I", "II"],key="mark_sub")
        else:
            class_name = st.selectbox("Select Class:", ["I", "II", "III"],key="mark_sub")

        # Fetch Subjects for Department and Class using view_subject()
        subjects = operation.dboperation.view_subjects(department_id, class_name)
        subject_dict = {subject[2]: subject[0] for subject in subjects}  # Subject Name: Subject ID

        subject_name = st.selectbox("Select Subject", list(subject_dict.keys()))
        if subject_name:
            subject_id = subject_dict[subject_name]

            # View Marks Subject-Wise
            if st.button("View Marks Subject-Wise"):
                students_marks_data = operation.dboperation.view_marks_class_department(department_id, class_name, subject_id)
                if students_marks_data:
                    st.write(f"Subject-Wise Marks for Department: {department_name}, Class: {class_name}, Subject: {subject_name}")
                    marks_df = pd.DataFrame(
                        students_marks_data,
                        columns=["Student ID", "Name","Class","Subject", "Quiz 1", "Quiz 2", "Quiz 3",
                                "Assignment 1", "Assignment 2", "Internal 1", "Internal 2", "Internal 3"]
                    )
                    st.dataframe(marks_df)
                else:
                    st.warning(f"No marks found for Subject: {subject_name}, Department: {department_name}, Class: {class_name}.")

        # Update Marks by Student ID and Subject ID
        if st.checkbox("Update Marks by Student ID"):
            students = operation.dboperation.view_students(department_id, class_name)
            student_ids = [student[0] for student in students]
            if student_ids:
                student_id = st.selectbox("Select Student ID to Update Marks:", student_ids)
                # Fetch existing marks for selected student
                marks_data = operation.dboperation.view_marks(student_id, subject_id)
                # print(marks_data)
                if marks_data:
                    _,_,_,quiz1, quiz2, quiz3, assignment1, assignment2, internal1, internal2, internal3 = marks_data

                    # Input fields to update marks
                    quiz1_new = st.number_input("Quiz 1", value=quiz1)
                    quiz2_new = st.number_input("Quiz 2", value=quiz2)
                    quiz3_new = st.number_input("Quiz 3", value=quiz3)
                    assignment1_new = st.number_input("Assignment 1", value=assignment1)
                    assignment2_new = st.number_input("Assignment 2", value=assignment2)
                    internal1_new = st.number_input("Internal 1", value=internal1)
                    internal2_new = st.number_input("Internal 2", value=internal2)
                    internal3_new = st.number_input("Internal 3", value=internal3)

                    # Update button
                    if st.button("Update Marks"):
                        operation.dboperation.update_marks(
                            student_id,
                            subject_id,
                            quiz1=quiz1_new,
                            quiz2=quiz2_new,
                            quiz3=quiz3_new,
                            assignment1=assignment1_new,
                            assignment2=assignment2_new,
                            internal1=internal1_new,
                           internal2= internal2_new,
                            internal3=internal3_new
                        )
                        st.success(f"Marks for Student ID {student_id} in Subject {subject_name} updated successfully.")
                else:
                    st.warning(f"No existing marks found for Student ID {student_id} in Subject {subject_name}.")
            else:
                st.warning(f"No students found for Department: {department_name}, Class: {class_name}.")

        # Delete Marks by Student ID
        if st.checkbox("Delete Marks by Student ID"):
            students = operation.dboperation.view_students(department_id, class_name)
            student_ids = [student[0] for student in students]
            if student_ids:
                student_id = st.selectbox("Select Student ID to Delete Marks:", student_ids)

                if st.button("Delete Marks"):
                    operation.dboperation.delete_marks(student_id)
                    st.success(f"Marks for Student ID {student_id} deleted successfully.")
            else:
                st.warning(f"No students found for Department: {department_name}, Class: {class_name}.")
     
    elif module == "admin data":
        st.subheader("admin")
        admin_id = st.text_input("enter the admin ID")
        if admin_id:
            operation.dboperation.add_admin(admin_id,'pass_admin')
            st.success(f"Admin {admin_id} added successfully")
        if st.checkbox("cache"):
            st.subheader("cache")
            st.table(operation.dboperation.view_cache())
            if st.checkbox("Updated by ID"):
                querys = operation.dboperation.view_cache()
                ids = [query[0] for query in querys]
                if ids:
                    id = st.selectbox("Select ID to Update Marks:", ids)
                    # Fetch existing marks for selected student
                    data = next((item for item in querys if item[0] == id), None)
                    if data:
                        new_question = st.text_input("enter the question",value=data[1])
                        new_answer = st.text_area("enter the question",value=data[2])
                        new_frequency = st.text_input("enter the question",value=data[3])
                        new_time = st.text_input("enter the question",value=data[4])
                        

                        # Update button
                        if st.button("Update Marks"):
                            operation.dboperation.update_cache(
                                id,
                                question=new_question,
                                answer=new_answer,
                                frequency=new_frequency,
                                timestamp=new_time
                            )
                            st.success(f"updated successfully.")
                    else:
                        st.warning(f"No existing ")
                else:
                    st.warning(f"No found")

            # Delete Marks by Student ID
            if st.checkbox("Delete by ID"):
                querys = operation.dboperation.view_cache()
                query_ids = [query[0] for query in querys]
                if query_ids:
                    id = st.selectbox("Select  ID to Delete :", query_ids)

                    if st.button("Delete Marks"):
                        operation.dboperation.delete_cache(id)
                        st.success(f"deleted successfully.")
                else:
                    st.warning(f"No found ")
        if st.checkbox("feedback"):
            st.subheader("feeedback")
            st.table(operation.dboperation.view_feedback())
            
            # Delete Marks by Student ID
            if st.checkbox("Delete by ID"):
                querys = operation.dboperation.view_feedback()
                query_ids = [query[0] for query in querys]
                if query_ids:
                    id = st.selectbox("Select  ID to Delete :", query_ids)

                    if st.button("Delete Marks"):
                        operation.dboperation.delete_feedback(id)
                        st.success(f"deleted successfully.")
                else:
                    st.warning(f"No found ")
     
    elif module == "Logout":
        st.session_state.authenticated = False
        st.session_state.page = "login"
        st.success("Logged out successfully!")
        st.rerun()
