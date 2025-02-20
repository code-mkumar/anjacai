import components
import streamlit as st

import components.admin
import components.guest
# import components.login
# import components.otpVerify
# import components.qrsetupp
import components.login
import components.otpVerify
import components.qrsetupp
import components.staff 

import components.student
# Main app
def app():
    # Initialize session state attributes
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "page" not in st.session_state:
        st.session_state.page = "guest"
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "multifactor" not in st.session_state:
        st.session_state.multifactor = None
    if "secret" not in st.session_state:
        st.session_state.secret = None
    if "role" not in st.session_state:
        st.session_state.role = None
    if "name" not in st.session_state:
        st.session_state.name = None
    if "id" not in st.session_state:
        st.session_state.id = None
    

    # Page navigation
    if st.session_state.page == "guest":
        components.guest.guest_page()
    elif st.session_state.page == "login":
        components.login.login_page()
    elif st.session_state.page == "qr_setup":
        components.qrsetupp.qr_setup_page()
    elif st.session_state.page == "otp_verification":
        components.otpVerify.otp_verification_page()
    elif st.session_state.page == "student":
        components.student.welcome_page()
    elif st.session_state.page == "staff":
        components.staff.staff_page()
    elif st.session_state.page == "admin":
        components.admin.admin_page()

# Run the app
if __name__ == "__main__":
    app()
