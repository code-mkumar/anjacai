import sqlite3
import operation.fileoperations
import os
import streamlit as st
def create_connection():
    
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/university.db"))
    return sqlite3.connect(db_path)
    
def create_tables():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = create_connection()
    cursor = conn.cursor()

    # SQL statements to create tables
    table_creation_queries = [
        """
        CREATE TABLE IF NOT EXISTS student_details (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dob DATE NOT NULL,
            department_id TEXT NOT NULL,
            class TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS student_mark_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            quiz1 FLOAT,
            quiz2 FLOAT,
            quiz3 FLOAT,
            assignment1 FLOAT,
            assignment2 FLOAT,
            internal1 FLOAT,
            internal2 FLOAT,
            internal3 FLOAT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS staff_details (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            designation TEXT NOT NULL,
            department_id TEXT NOT NULL,
            password TEXT NOT NULL DEFAULT 'pass_staff',
            mfa BOOLEAN DEFAULT 0,
            secd TEXT DEFAULT 'NONE',
            phone_no INTEGER NOT NULL,
            email TEXT NOT NULL 
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS department_details (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            grad_level TEXT NOT NULL,
            phone TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS admin_details (
            id TEXT PRIMARY KEY,
            password TEXT NOT NULL DEFAULT 'pass_admin',
            mfa BOOLEAN DEFAULT 0,
            secd TEXT DEFAULT ''
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            message TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS subject (
            id TEXT PRIMARY KEY,
            department_id TEXT NOT NULL,
            name TEXT NOT NULL,
            class TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS timetable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT NOT NULL,
            time TEXT NOT NULL,
            subject TEXT NOT NULL,
            class TEXT NOT NULL,
            department_id TEXT NOT NULL
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_student_details ON student_details (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_student_mark_details ON student_mark_details (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_staff_details ON staff_details (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_details ON admin_details (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_feedback ON feedback (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_timetable ON timetable (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_subject ON subject (id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_department ON department_details (id);
        """
    ]

    # Execute all queries
    for query in table_creation_queries:
        cursor.execute(query)

    # Commit and close the connection
    conn.commit()
    conn.close()


# function to get the user role
def get_role(user_id):
    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        role = ''
        tables = ['student_details', 'staff_details', 'admin_details']
        
        for table in tables:
            cursor.execute(f'SELECT 1 FROM {table} WHERE id = ?', (user_id,))
            if cursor.fetchone():  # Checks if any row is returned
                role = table
                break
        
        return role
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# funtcion to verify the user password
def check_user(user_id, password, role):
    from datetime import date
    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        query = ""
        dob=''
        # Validate role to prevent SQL injection
        if role not in ['student_details', 'staff_details', 'admin_details']:
            raise ValueError("Invalid role specified.")
        if role == 'student_details':
            dob = date.fromisoformat(password)
        # Construct query based on role
        if role == 'student_details':
            query = f'SELECT * FROM {role} WHERE id = ? AND dob = ?'
            cursor.execute(query, (user_id, dob))
        else:
            query = f'SELECT * FROM {role} WHERE id = ? AND password = ?'
            cursor.execute(query, (user_id, password))
        data = cursor.fetchone()
        # print(query)
        # Execute the query
        # print(data)
        return data
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    except ValueError as ve:
        print(f"Value error: {ve}")
        return None
    finally:
        if conn:
            conn.close()

# enabling or disabling the mfa
def mfa_update(user_id, role, status):
    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()

        # Validate role to prevent SQL injection
        if role not in ['staff_details', 'admin_details']:
            raise ValueError("Invalid role specified.")

        # Validate status as integer or boolean (convert to 1/0 for boolean)
        if not isinstance(status, (int, bool)):
            raise ValueError("Invalid status type. Must be an integer or boolean.")
        status = int(status)

        # Construct and execute the query
        query = f"UPDATE {role} SET mfa = ? WHERE id = ?"
        cursor.execute(query, (status, user_id))
        conn.commit()
        
        return "success"
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "failed"
    except ValueError as ve:
        print(f"Value error: {ve}")
        return "failed"
    finally:
        if conn:
            conn.close()

def get_info_of_mfa(user_id,role):
    conn=None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        if role not in ['staff_details','admin_details']:
            raise ValueError("Invalid role specified.")
        query = f"SELECT mfa ,secd FROM {role} WHERE id = ?"
        cursor.execute(query,(user_id,))
        data = cursor.fetchone()
        return True if data[0] == 1 else False
    except Exception as e:
        print(e)

def get_mfa_and_serectcode(user_id,role):
    conn=None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        if role not in ['staff_details','admin_details']:
            raise ValueError("Invalid role specified.")
        query = f"SELECT mfa ,secd FROM {role} WHERE id = ?"
        cursor.execute(query,(user_id,))
        data = cursor.fetchone()
        return data
    except Exception as e:
        print(e)

#function to update the serect code
def serectcode_update(user_id, secret, role):
    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()

        # Validate role to prevent SQL injection
        if role not in ['staff_details', 'admin_details']:
            raise ValueError("Invalid role specified.")

        # Validate secret as a string (assuming it's a secret code, not a boolean or integer)
        if not isinstance(secret, str) or len(secret) == 0:
            raise ValueError("Invalid secret type. Must be a non-empty string.")

        # Construct and execute the query with conditional MFA check
        query = f"UPDATE {role} SET secd = ? WHERE id = ? AND mfa = 1"
        cursor.execute(query, (secret, user_id))
        conn.commit()
        
        return "success"
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "failed"
    except ValueError as ve:
        print(f"Value error: {ve}")
        return "failed"
    finally:
        if conn:
            conn.close()

#remove and disable the mfa
def clear_mfa(user_id, role):
    conn = None
    try:
        # Validate role to prevent SQL injection
        if role not in ['staff_details', 'admin_details']:
            raise ValueError("Invalid role specified.")
        
        # Establish the database connection
        conn = create_connection()
        cursor = conn.cursor()

        # Construct and execute the query to clear MFA
        query = f"UPDATE {role} SET secd = 'None', mfa = 0 WHERE id = ? AND mfa = 1"
        cursor.execute(query, (user_id,))
        conn.commit()

        return "success"
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "failed"
    except ValueError as ve:
        print(f"Value error: {ve}")
        return "failed"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "failed"
    finally:
        if conn:
            conn.close()


#user details by user id 
def get_user_details(user_id):
    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        tables = ['student_details', 'staff_details', 'admin_details']
        data=''
        for table in tables:
            cursor.execute(f'SELECT * FROM {table} WHERE id = ?', (user_id,))
            data = cursor.fetchone()
        return data
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()


def read_sql_query(sql):
    try:
        # st.write(sql)
        conn = create_connection()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.commit()
        conn.close()
        # st.write(rows)
        return rows
    except Exception as e:
        #print(sql)
        print( "sql error:",e)
        return f"SQLite error: {e}"


def change_pass(password,user_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE staff_details SET password = ? WHERE id = ?", (password, user_id))
    conn.commit()
    conn.close()

# incrementing the class and delete the final 
def incrementing_class():
    conn = None
    try:
        # Establish the database connection
        conn = create_connection()
        cursor = conn.cursor()

        # Delete students who are in their final year (class III)
        query_delete_final_year = "DELETE FROM student_details WHERE class = 'III'"
        cursor.execute(query_delete_final_year)

        # Increment class from I to II
        query_update_I_to_II = "UPDATE student_details SET class = 'II' WHERE class = 'I'"
        cursor.execute(query_update_I_to_II)

        # Increment class from II to III
        query_update_II_to_III = "UPDATE student_details SET class = 'III' WHERE class = 'II'"
        cursor.execute(query_update_II_to_III)

        # Commit the changes to the database
        conn.commit()
        return "success"
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "failed"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "failed"
    finally:
        if conn:
            conn.close()

# Function to add a new admin
def add_admin(admin_id,password, mfa=False, secd=''):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO admin_details (id,password, mfa, secd)
    VALUES (?, ?, ?, ?);
    """
    cursor.execute(query, (admin_id,password, mfa, secd))
    conn.commit()
    conn.close()
    print("Admin added successfully.")

# Function to view admins
def view_admins():
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM admin_details;"
    cursor.execute(query)
    admins = cursor.fetchall()
    conn.close()
    return admins

# Function to update an admin's password
def update_admin_password(admin_id, new_password):
    conn = create_connection()
    cursor = conn.cursor()
    query = "UPDATE admin_details SET password = ? WHERE id = ?;"
    cursor.execute(query, (new_password, admin_id))
    conn.commit()
    conn.close()
    print(f"Admin {admin_id} password updated.")

# Function to delete an admin
def delete_admin(admin_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM admin_details WHERE id = ?;"
    cursor.execute(query, (admin_id,))
    conn.commit()
    conn.close()
    print(f"Admin {admin_id} deleted.")

# Function to add a new student
def add_student(student_id,name, date_of_birth, department_id, class_name):
    conn = create_connection()
    cursor=conn.cursor()
    query = """
    INSERT INTO student_details (id,name, dob, department_id, class)
    VALUES (?, ?, ?, ?, ?);
    """
    cursor.execute(query, (student_id,name, date_of_birth, department_id, class_name,))
    conn.commit()
    conn.close()
    print(f"Student {name} added successfully.")

# Function to view all students
def view_students(department_id,class_name):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM student_details where department_id = ? AND class =? ;"
    cursor.execute(query,(department_id,class_name,))
    students = cursor.fetchall()
    conn.close()
    return students

# Function to view a specific student by ID
def view_student(student_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM student_details WHERE id = ?;"
    cursor.execute(query, (student_id,))
    student = cursor.fetchone()
    conn.close()
    return student

# Function to update student details
def update_student(student_id, name=None, date_of_birth=None, department_id=None, class_name=None):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    UPDATE student_details
    SET name = ?, dob = ?, department_id = ?, class = ?
    WHERE id = ?;
    """
    cursor.execute(query, (name, date_of_birth, department_id, class_name, student_id))
    conn.commit()
    conn.close()
    print(f"Student {student_id} details updated.")

# Function to delete a student by ID
def delete_student(student_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM student_details WHERE id = ?;"
    cursor.execute(query, (student_id,))
    conn.commit()
    conn.close()
    print(f"Student {student_id} deleted.")


# Function to add a new staff
def add_staff(staff_id , name, designation, department_id, password="pass_staff", mfa=False, secd=None, phone_no='', email=''):
    conn = create_connection()
    st.write("Within add_staff method:")
    st.write(department_id)
    cursor = conn.cursor()
    query = """
    INSERT INTO staff_details (id , name, designation, department_id, password, mfa, secd, phone_no, email)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?,?);
    """
    cursor.execute(query, (staff_id,name, designation, department_id, password, mfa, secd, phone_no, email))
    conn.commit()
    conn.close()
    print("Staff added successfully.")

# Function to view staff
def view_staffs(department_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM staff_details where department_id =?;"
    cursor.execute(query,(department_id,))
    staff = cursor.fetchall()
    conn.close()
    return staff

def view_staff(user_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM staff_details where id =?;"
    cursor.execute(query,(user_id,))
    staff = cursor.fetchall()
    conn.close()
    return staff

# Function to update staff details
def update_staff(staff_id, name=None, designation=None, department_id=None, password=None, mfa=None, secd=None, phone_no=None, email=None):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    UPDATE staff_details
    SET name = ?, designation = ?, department_id = ?, password = ?, mfa = ?, secd = ?, phone_no = ?, email = ?
    WHERE id = ?;
    """
    cursor.execute(query, (name, designation, department_id, password, mfa, secd, phone_no, email, staff_id))
    conn.commit()
    conn.close()
    print(f"Staff {staff_id} details updated.")

# Function to delete a staff member
def delete_staff(staff_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM staff_details WHERE id = ?;"
    cursor.execute(query, (staff_id,))
    conn.commit()
    conn.close()
    print(f"Staff {staff_id} deleted.")

def view_cache():
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM cache;"
    cursor.execute(query)
    staff = cursor.fetchall()
    conn.close()
    return staff

# Function to update staff details
def update_cache(id, question=None, answer=None, frequency=None, timestamp=None):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    UPDATE cache
    SET question = ?, answer = ?, frequency = ?, timestamp = ?
    WHERE id = ?;
    """
    cursor.execute(query, (question,answer,frequency,timestamp,id))
    conn.commit()
    conn.close()
    print(f" {id} details updated.")

# Function to delete a staff member
def delete_cache(id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM cache WHERE id = ?;"
    cursor.execute(query, (id,))
    conn.commit()
    conn.close()
    print(f"Staff {id} deleted.")

# Function to add a new department
def add_department(department_id,name, grad_level, phone):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO department_details (id,name, grad_level, phone)
    VALUES (?, ?, ?,?);
    """
    cursor.execute(query, (department_id,name, grad_level, phone))
    conn.commit()
    conn.close()
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../files/{name}_department.txt"))
    filename = f"{name}_department.txt"
    department_info = f"Department: {name}\nGraduation Level: {grad_level}\nPhone: {phone}"  
    # Write department details to the file
    operation.fileoperations.write_to_file(file_path, department_info)
    
    print("Department added successfully.")

# Function to view departments
def view_departments():
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM department_details;"
    cursor.execute(query)
    departments = cursor.fetchall()
    conn.close()
    return departments

# Function to update department details
def update_department(department_id, name=None, grad_level=None, phone=None):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    UPDATE department_details
    SET name = ?, grad_level = ?, phone = ?
    WHERE id = ?;
    """
    cursor.execute(query, (name, grad_level, phone, department_id))
    conn.commit()
    conn.close()
    print(f"Department {department_id} details updated.")
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../files/{name}_department.txt"))
    filename = f"{name}_department.txt"
    department_info = f"Department: {name}\nGraduation Level: {grad_level}\nPhone: {phone}"  
    # Write department details to the file
    operation.fileoperations.write_to_file(file_path, "")
    

# Function to delete a department
def delete_department(department_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM department_details WHERE id = ?;"
    cursor.execute(query, (department_id,))
    conn.commit()
    conn.close()
    print(f"Department {department_id} deleted.")


# Function to add a new subject
def add_subject(subject_id,department_id,name,class_name):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO subject (id,department_id, name,class)
    VALUES (?, ?,?,?);
    """
    cursor.execute(query, (subject_id,department_id, name,class_name))
    conn.commit()
    conn.close()
    print("Subject added successfully.")

# Function to view subjects
def view_subjects(department_id,class_name):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM subject where department_id =?AND class =?;"
    cursor.execute(query,(department_id,class_name,))
    subjects = cursor.fetchall()
    conn.close()
    return subjects 

def view_subjects_department(department_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM subject where department_id =?;"
    cursor.execute(query,(department_id,))
    subjects = cursor.fetchall()
    conn.close()
    return subjects

# Function to update subject details
def update_subject(subject_id, department_id=None, name=None,class_name=None):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    UPDATE subject
    SET department_id = ?, name = ? , class = ?
    WHERE id = ?;
    """
    cursor.execute(query, (department_id, name, subject_id,class_name))
    conn.commit()
    conn.close()
    print(f"Subject {subject_id} updated.")

# Function to delete a subject
def delete_subject(subject_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM subject WHERE id = ?;"
    cursor.execute(query, (subject_id,))
    conn.commit()
    conn.close()
    print(f"Subject {subject_id} deleted.")


# Function to add a new timetable entry
def add_timetable(day, time, subject, class_name, department_id,staff_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO timetable (day, time, subject, class, department_id,staff_id)
    VALUES (?, ?, ?, ?, ?,?);
    """
    cursor.execute(query, (day, time, subject, class_name, department_id,staff_id))
    conn.commit()
    conn.close()
    print("Timetable entry added successfully.")

# Function to view timetable entries
def view_timetable(department_id,class_name):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM timetable where department_id =? AND class=?;"
    cursor.execute(query,(department_id,class_name,))
    timetable = cursor.fetchall()
    conn.close()
    return timetable

# Function to update a timetable entry
def update_timetable(timetable_id, day=None, time=None, subject=None, class_name=None, department_id=None,staff_id=None):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    UPDATE timetable
    SET day = ?, time = ?, subject = ?, class = ?, department_id = ?,staff_id=?
    WHERE id = ?;
    """
    cursor.execute(query, (day, time, subject, class_name, department_id,staff_id, timetable_id))
    conn.commit()
    conn.close()
    print(f"Timetable entry {timetable_id} updated.")

# Function to delete a timetable entry
def delete_timetable(timetable_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM timetable WHERE id = ?;"
    cursor.execute(query, (timetable_id,))
    conn.commit()
    conn.close()
    print(f"Timetable entry {timetable_id} deleted.")
    
    
  

# Function to add marks (quiz, assignment, internal) for a specific cycle (1, 2, or 3)
def add_marks(student_id, subject_id, cycle, quiz=None, assignment=None, internal=None):
    if cycle not in ['1', '2', '3']:
        print("Invalid cycle. Please choose from 1, 2, or 3.")
        return

    # Create column names dynamically based on the cycle
    quiz_column = f"quiz{cycle}"
    assignment_column = f"assignment{cycle}" if cycle != '3' else None
    internal_column = f"internal{cycle}"

    conn = create_connection()
    cursor = conn.cursor()

    # Ensure the row exists before updating
    cursor.execute("""
    INSERT OR IGNORE INTO student_mark_details (student_id, subject_id)
    VALUES (?, ?);
    """, (student_id, subject_id))

    # Update the row based on the cycle
    if cycle == 3:
        query = f"""
        UPDATE student_mark_details
        SET {quiz_column} = ?, {internal_column} = ?
        WHERE student_id = ? AND subject_id = ?;
        """
        cursor.execute(query, (quiz, internal, student_id, subject_id))
    else:
        query = f"""
        UPDATE student_mark_details
        SET {quiz_column} = ?, {assignment_column} = ?, {internal_column} = ?
        WHERE student_id = ? AND subject_id = ?;
        """
        cursor.execute(query, (quiz, assignment, internal, student_id, subject_id))

    conn.commit()
    conn.close()
    print(f"Marks added/updated for student ID {student_id} and subject ID {subject_id} in cycle {cycle}.")



# Function to view marks (quiz, assignment, internal) for a specific cycle (1, 2, or 3)
def view_marks(student_id, subject_id):
    

    conn = create_connection()
    cursor = conn.cursor()

   
    query = f"""
        SELECT *
        FROM student_mark_details
        WHERE student_id = ? AND subject_id = ?;
        """
    cursor.execute(query, (student_id, subject_id))

    marks = cursor.fetchone()
    conn.close()

    if marks:
        return marks
    else:
        print(f"No marks found for student ID {student_id}, subject ID {subject_id}.")
        return None


# Function to update marks (quiz, assignment, internal) for a specific cycle (1, 2, or 3)
def update_marks(student_id,subject_id, quiz1=None, assignment1=None, internal1=None,quiz2=None, assignment2=None, internal2=None,quiz3=None,internal3=None):

    conn = create_connection()
    cursor = conn.cursor()

    
        # For cycle 1 and 2, exclude assignment column
    query = f"""
        UPDATE student_mark_details
        SET quiz1 = ?, internal1 = ?,assignment1=?,quiz2 = ?, internal2 = ?,assignment2=?,quiz3= ?, internal3 = ?
        WHERE id = ? and subject_id=?;
        """
    cursor.execute(query, (quiz1, internal1,assignment1,quiz2, internal2,assignment2,quiz3, internal3, student_id,subject_id))

    conn.commit()
    conn.close()
    print(f"Marks for cycle  updated successfully for student ID {student_id} for {subject_id}.")

def view_marks_class_department(department_id, class_name, subject_id):
    conn = create_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            sd.id AS student_id,
            sd.name AS student_name,
            sd.class AS class,
            s.name AS subject_name,
            sm.quiz1,
            sm.quiz2,
            sm.quiz3,
            sm.assignment1,
            sm.assignment2,
            sm.internal1,
            sm.internal2,
            sm.internal3
        FROM 
            student_details sd
        JOIN 
            student_mark_details sm ON sd.id = sm.student_id
        JOIN 
            subject s ON sm.subject_id = s.id
        WHERE 
            sd.department_id = ? AND sd.class = ? AND s.id = ?;
    """
    cursor.execute(query, (department_id, class_name, subject_id))
    results = cursor.fetchall()
    conn.close()

    return results

# Function to delete marks (quiz, assignment, internal) for a specific cycle (1, 2, or 3)
def delete_marks(student_id):
    conn = create_connection()
    cursor = conn.cursor()

    # Delete the marks for the selected cycle
   
    query = f"""
        DELETE FROM student_mark_details WHERE student_id = ?;
        """
    cursor.execute(query, (student_id,))

    conn.commit()
    conn.close()
    print(f"Marks for deleted successfully for student ID {student_id}.")


# Function to add feedback
def add_feedback( user_id, name, message):
    """
    Adds a feedback entry to the database.

    :param db_path: Path to the SQLite database file.
    :param user_id: ID of the user providing the feedback.
    :param name: Name of the user providing the feedback.
    :param message: Feedback message.
    """
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_id, name, message)
            VALUES (?, ?, ?)
        """, (user_id, name, message))
        conn.commit()
        print("Feedback added successfully!")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

# Function to delete feedback
def delete_feedback( feedback_id):
    """
    Deletes a feedback entry from the database.

    :param db_path: Path to the SQLite database file.
    :param feedback_id: ID of the feedback entry to delete.
    """
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM feedback WHERE id = ?
        """, (feedback_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print("Feedback deleted successfully!")
        else:
            print("No feedback found with the provided ID.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

def view_feedback():
    

    conn = create_connection()
    cursor = conn.cursor()

   
    query = f"""
        SELECT *
        FROM feedback;
        """
    cursor.execute(query)

    marks = cursor.fetchall()
    conn.close()

    if marks:
        return marks
    else:
        print(f"No  found .")
        return None
