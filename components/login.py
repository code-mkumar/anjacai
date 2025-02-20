import streamlit as st
import operation
import operation.dboperation
#import operation.dboperation
from datetime import date
def login_page():
    st.set_page_config(page_title="Login")
    st.title("Login")
    user_id = st.text_input("User ID").upper()
    # print(user_id)
    # Define the range of dates
    min_date = date(2000, 1, 1)  # Minimum date
    max_date = date(2050, 12, 31)  # Maximum date
    role = operation.dboperation.get_role(user_id)
    dob=''
    password=''
    if role == "student_details":
        dob = st.date_input("Date of Birth",min_value=min_date,max_value=max_date)
        print(dob)
    elif not role == "student_details":
        password = st.text_input("Password", type="password")
    if "role" not in st.session_state:
        st.session_state.role = ''
    st.session_state.role = role
    if st.button("Login"):
        if user_id and (password or dob):
            # st.session_state.authenticated = True
            st.session_state.user_id = user_id
            auth_param = dob.isoformat() if role == "student_details" else password
            # print(auth_param)
            user = operation.dboperation.check_user(user_id, auth_param, role)
            #user = operation.dboperation.check_user(user_id,password,role)
            if user:    
            # st.session_state.multifactor = user[8]  # Multifactor column
            # st.session_state.secret = user[9]  # Secret code column
                st.success("Login successful!")
                # if st.session_state.multifactor == 1:
                #     st.session_state.page = "otp_verification"  # Direct to OTP verification if MFA is enabled
                #     st.rerun()
                # else:
                #     if st.session_state.secret == "None":
                #         st.session_state.page = "qr_setup"  # If MFA is not enabled, show QR setup
                #         st.rerun()
                #     else:
                #         st.session_state.page = "otp_verification"  # Otherwise show OTP verification
                #         st.rerun()
                if role == "student_details":
                    st.session_state.page = "student"
                    st.rerun()
                elif role == "staff_details" or role == "admin_details":
                    print(role)
                    if operation.dboperation.get_info_of_mfa(user_id,role):
                        mfa,secd= operation.dboperation.get_mfa_and_serectcode(user_id,role)
                       
                        if "secret" not in st.session_state:
                            st.session_state.secret=''
                        if mfa == 1 and secd==None or len(secd)==0 :
                            st.session_state.page ="qr_setup"
                            st.rerun()
                        elif mfa == 1 and  secd:
                            st.session_state.secret=secd
                            st.session_state.page ="otp_verification"
                            st.rerun()
                    else:
                        if role == "admin_details":
                            st.session_state.page ="admin"
                            st.rerun()
                        else:
                            st.session_state.page ="staff"
                            st.rerun()
                elif role == "admin_details":
                    st.session_state.page="admin"
                    st.rerun() 
                st.rerun()        
            else:
                st.error("Invalid credentials.")
    if st.button("←--"):
        st.session_state.page = "guest"
        st.rerun()

