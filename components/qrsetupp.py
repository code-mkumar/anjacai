import streamlit as st
import operation
# import operation.dboperation
# import operation.fileoperations
import operation.dboperation
import operation.qrsetter
import operation.secretcode

def qr_setup_page():
    # st.set_page_config(page_title="QRcode")
    st.title("Setup Multifactor Authentication")
    user_id = st.session_state.user_id
    role= st.session_state.role
    if st.session_state.secret == None:
        # Generate a new secret
        secret = operation.secretcode.generate_secret_code(user_id)
        st.session_state.secret = secret
    else:
        secret = st.session_state.secret

    # Display QR Code
    qr_code_stream = operation.qrsetter.generate_qr_code(user_id, secret)
    st.image(qr_code_stream, caption="Scan this QR code with your authenticator app.")
    st.write(f"Secret Code: `{secret}` (store this securely!)")
    operation.dboperation.serectcode_update(user_id,secret,role)  # Update MFA status in the database
    # Immediate OTP verification
    otp = st.text_input("Enter OTP from Authenticator App", type="password")
    if st.button("Verify OTP"):
        # secret, role, name = get_user_details(st.session_state.user_id)
        if operation.qrsetter.verify_otp(st.session_state.secret, otp):
            if role == "staff_details":
                st.session_state.page = "staff"
                st.rerun()
            if role == "admin_details":
                st.session_state.page = "admin"
                st.rerun()
            st.success("Multifactor authentication is now enabled.")
            # st.session_state.page = "welcome"
            # st.rerun()
        else:
            st.error("Invalid OTP. Try again.")
