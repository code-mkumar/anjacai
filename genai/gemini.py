import google.generativeai as genai
import streamlit as st
genai.configure(api_key='AIzaSyD3WqHberJDYyzXkmY1zKaoqd5uCJZDetI')
model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(combined_prompt,data):
    import re
    print(str(data))
    response = model.generate_content(combined_prompt+"this is the information about the user "+str(data))
    # print(response)
    query=response.text
    return query
    # # Add user_id for personal queries if not already included
    # if "my" in combined_prompt.lower() and "id" not in query.lower():
    #     query = query.strip(";") + f" WHERE id='{st.session_state.id}';"

    # # Remove unnecessary user_id filters for general queries
    # general_contexts = ["department names", "course names", "college history", "programmes of study"]
    # if any(context in combined_prompt.lower() for context in general_contexts):
    #     query = re.sub(r"WHERE\s+user_id\s*=\s*['\"]\w+['\"]", "", query, flags=re.IGNORECASE)

    # return query
    # id = st.session_state.id
    # try:
    #     final = model.generate_content(f"{response.text} if any user_id word found in this statement replace with {id}")
    #     #final=model.generate_content(response.text)
    # except:
    #     return "please contact to the staff or admin"
    # return final.text
    # # if not response or 'candidates' not in response:
    # #     return "The model could not generate a valid response. Please try again."

    # # candidate_content = response.candidates[0].content.parts[0].text
    # # return candidate_content if candidate_content else "No valid content returned from the candidate."
