import streamlit as st
import operation
# import operation.dboperation
# import operation.fileoperations
import operation.dboperation
import operation.qrsetter
def otp_verification_page():
    # st.set_page_config(page_title="verify")
    st.title("Verify OTP")
    user_id = st.session_state.user_id
    role = st.session_state.role
    _ ,secd = operation.dboperation.get_mfa_and_serectcode(user_id,role)
    # secret, role, name = operation.dboperation.get_user_details(user_id)
    otp = st.text_input("Enter OTP", type="password")
    if st.button("Verify"):
        if operation.qrsetter.verify_otp(secd, otp):
            st.success("OTP Verified! Welcome.")
            if role == "staff_details":
                st.session_state.page = "staff"
                st.rerun()
            if role == "admin_details":
                st.session_state.page = "admin"
                st.rerun()
            
        else:
            st.error("Invalid OTP. Try again.")
