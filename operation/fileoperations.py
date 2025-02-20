import os
# def read_student_files():
#     with open("student_role.txt", "r") as role_file:
#         role_content = role_file.read()
#     with open("student_sql.txt", "r") as sql_file:
#         sql_content = sql_file.read()
#     return role_content, sql_content

# def read_default_files():
#     with open("default.txt", "r") as role_file:
#         role_content = role_file.read()
#     with open("default_sql.txt", "r") as sql_file:
#         sql_content = sql_file.read()
#     return role_content, sql_content

# def read_staff_files():
#     with open("staff_role.txt", "r") as role_file:
#         role_content = role_file.read()
#     with open("staff_sql.txt", "r") as sql_file:
#         sql_content = sql_file.read()
#     return role_content, sql_content

# def read_admin_files():
#     with open("admin_role.txt", "r") as role_file:
#         role_content = role_file.read()
#     with open("admin_sql.txt", "r") as sql_file:
#         sql_content = sql_file.read()
#     return role_content, sql_content

# def write_to_file(filename, data=''):
#     """
#     Writes data to a file, overwriting the existing content.
#     :param filename: The file to write to (student.txt or staff.txt).
#     :param data: The data to write (a list of strings).
#     """
#     # Construct the absolute path for the `files` directory
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one level
#     files_dir = os.path.join(base_dir, "files")  # Path to the `files` directory

#     # Ensure the `files` directory exists
#     os.makedirs(files_dir, exist_ok=True)

#     # Full path to the target file
#     file_path = os.path.join(files_dir, f"{filename}.txt")
#     with open(file_path, "w") as file:
#         for line in data:
#             file.write(line + "\n")
#     print(f"Data successfully written to {filename}.")

# def append_to_file(filename, data):
#     """
#     Appends data to a file.
#     :param filename: The file to append to (student.txt or staff.txt).
#     :param data: The data to append (a list of strings).
#     """
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one level
#     files_dir = os.path.join(base_dir, "files")  # Path to the `files` directory

#     # Ensure the `files` directory exists
#     os.makedirs(files_dir, exist_ok=True)

#     # Full path to the target file
#     file_path = os.path.join(files_dir, {filename})
#     with open(file_path, "a") as file:
#         for line in data:
#             file.write(line + "\n")
#     print(f"Data successfully appended to {filename}.")


# def read_from_file(filename):
#     """
#     Reads data from a file and returns it as a list of strings.
#     :param filename: The file to read from (student.txt or staff.txt).
#     :return: A list of strings (one per line in the file).
#     """
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one level
#     files_dir = os.path.join(base_dir, "files")  # Path to the `files` directory

#     # Ensure the `files` directory exists
#     os.makedirs(files_dir, exist_ok=True)

#     # Full path to the target file
#     file_path = os.path.join(files_dir, {filename})
#     with open(file_path, "r") as file:
#         data = file.readlines()
#     return [line.strip() for line in data]  # Remove newlines from each line
import os

def write_to_file(filename, data=''):
    """
    Writes data to a file, overwriting the existing content.
    :param filename: The file to write to (student.txt or staff.txt).
    :param data: The data to write (a list of strings).
    """
    # Construct the absolute path for the `files` directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one level
    files_dir = os.path.join(base_dir, "files")  # Path to the `files` directory

    # Ensure the `files` directory exists
    os.makedirs(files_dir, exist_ok=True)

    # Full path to the target file
    file_path = os.path.join(files_dir, f"{filename}")
    with open(file_path, "w") as file:
        # for line in data:
        #     file.write(line)
        file.write(data)
    print(f"Data successfully written to {filename}.")

def append_to_file(filename, data):
    """
    Appends data to a file.
    :param filename: The file to append to (student.txt or staff.txt).
    :param data: The data to append (a list of strings).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one level
    files_dir = os.path.join(base_dir, "files")  # Path to the `files` directory

    # Ensure the `files` directory exists
    os.makedirs(files_dir, exist_ok=True)

    # Full path to the target file
    file_path = os.path.join(files_dir, f"{filename}.txt")
    with open(f"../files/{filename}", "a") as file:
        for line in data:
            file.write(line + "\n")
    print(f"Data successfully appended to {filename}.")

def read_from_file(filename):
    """
    Reads data from a file and returns it as a list of strings.
    :param filename: The file to read from (student.txt or staff.txt).
    :return: A list of strings (one per line in the file).
    """
    # print("file name",filename)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one level
    files_dir = os.path.join(base_dir, "files")  # Path to the `files` directory

    # Ensure the `files` directory exists
    os.makedirs(files_dir, exist_ok=True)
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../files/"))
    # Full path to the target file
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r",encoding='utf-8') as file:
        data = file.readlines()
    #return [line.strip() for line in data]  # Remove newlines from each line
    return data

def file_to_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_content = "".join([page.extract_text() for page in pdf_reader.pages])
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        from docx import Document
        doc = Document(uploaded_file)
        file_content = "\n".join([p.text for p in doc.paragraphs])
    else:
        file_content = uploaded_file.read().decode('utf-8')
    
    return file_content
