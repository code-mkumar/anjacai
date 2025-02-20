# import streamlit as st

# # sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
# selected = st.feedback("thumbs")
# # if selected is not None:
# #     st.markdown(f"You selected: {sentiment_mapping[selected]}")

# def chatbot():
#     responses = {
#         "current department": ["my", "our"],
#         "UGTAMIL": ["tamil department ug", "department of tamil", "bsc tamil", "ug tamil","tamil department"],
#         "UGHINDI": ["hindi department ug", "department of hindi", "ba hindi", "ug hindi","hindi department"],
#         "PART2ENG": ["english department ug", "department of english", "ba english", "ug english","english department"],
#         "UGMAT": ["mathematics department ug", "department of mathematics", "bsc mathematics", "ug mathematics","maths department"],
#         "UGPHY": ["physics department ug", "department of physics", "bsc physics", "ug physics","phusics department"],
#         "UGCHE": ["chemistry department ug", "department of chemistry", "bsc chemistry", "ug chemistry","chemistry department"],
#         "UGBOT": ["botany department ug", "department of botany", "bsc botany", "ug botany","botany department"],
#         "UGZOO": ["zoology department ug", "department of zoology", "bsc zoology", "ug zoology","zoology department"],
#         "UGPHS": ["physical education department ug", "department of physical education", "bsc physical education", "ug physical education","phs","physical education department"],
#         "UGECO": ["economics department ug", "department of economics", "ba economics", "ug economics","economic department","economic department"],
#         "UGCOM": ["commerce department ug", "department of commerce", "bcom commerce", "ug commerce","bcom","commerce general","commerce department"],
#         "UGBBAR": ["business administration department ug", "department of business administration", "bba business administration", "ug business administration","bba regular"],
#         "UGMICRO": ["microbiology department ug", "department of microbiology", "bsc microbiology", "ug microbiology","microbilogy department"],
#         "PGMICRO": ["microbiology department pg", "department of microbiology", "msc microbiology", "pg microbiology"],
#         "UGBIOTECH": ["biotechnology department ug", "department of biotechnology", "bsc biotechnology", "ug biotechnology","biotech","bio-tech","biotecnology department"],
#         "PGBIOTECH": ["biotechnology department pg", "department of biotechnology", "msc biotechnology", "pg biotechnology"],
#         "UGVISCOM": ["visual communication department ug", "department of visual communication", "bsc visual communication", "ug visual communication","viscom","visual communication department"],
#         "UGCSSF": ["computer science department ug sf", "department of computer science sf", "bsc computer science sf", "ug computer science sf","computer science self"],
#         "UGBCA": ["computer application department ug", "department of computer application", "bca computer applications", "ug computer applications","bca","microbilogy department"],
#         "UGPHSSF": ["physical education department ug sf", "department of physical education sf", "bsc physical education sf", "ug physical education sf","phs-sf","phs sf","phs-self","physical education department self"],
#         "UGENG": ["english department ug", "department of english", "ba english", "ug english","english department","eng"],
#         "UGCCS": ["commerce corporate secretaryship department ug", "department of commerce corporate secretaryship", "bcom corporate secretaryship", "ug corporate secretaryship","corporate secretaryship department"],
#         "PGCOM": ["commerce department pg", "department of commerce", "mcom commerce", "pg commerce","mcom"],
#         "UGBBASF": ["business administration department ug sf", "department of business administration sf", "bba business administration sf", "ug business administration sf","bba sf","bba self","bba-sf"],
#         "PGTAMIL": ["tamil department pg", "department of tamil", "ma tamil", "pg tamil"],
#         "PGMAT": ["mathematics department pg", "department of mathematics", "msc mathematics", "pg mathematics","msc maths"],
#         "PGPHY": ["physics department pg", "department of physics", "msc physics", "pg physics"],
#         "PGCHE": ["chemistry department pg", "department of chemistry", "msc chemistry", "pg chemistry"],
#         "PGBOT": ["botany department pg", "department of botany", "msc botany", "pg botany"],
#         "PGZOO": ["zoology department pg", "department of zoology", "msc zoology", "pg zoology"],
#         "PGCS": ["computer science department pg", "department of computer science", "msc computer science", "pg computer science","msc cs"],
#         "PGMCA": ["computer application department pg", "department of computer application", "mca computer applications", "pg computer applications","mca"],
#         "PGECO": ["economics department pg", "department of economics", "ma economics", "pg economics"],
#         "UGCSR": ["computer science department ug", "department of computer science", "bsc computer science", "ug computer science","bsc cs","computer department","computer science department"],
#         "UGCOMEC": ["commerce ug-ca & ec department", "department of commerce ug-ca & ec", "bcom commerce ug-ca & ec", "ug commerce ug-ca & ec"],
#         "UGCPA": ["commerce professional accounting department ug sf", "department of commerce professional accounting sf", "bcom professional accounting", "ug professional accounting"]
#     }
#     def get_response(user_input):
#         for key, values in responses.items():
#             for phrase in values:
#                 if phrase in user_input:
#                     return key
#         return "Sorry, I don't understand that."
    
#     while True:
#         user_input = input("You: ").lower()
#         response = get_response(user_input)
#         print("Chatbot:", response)
        
#         if user_input == "bye":
#             break

# chatbot()

# chunks = ["chunk1", "chunk2", "chunk3"]
import re
import streamlit as st
import os

# Folder Path
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./files/"))

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

