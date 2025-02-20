import pyotp

def generate_secret_code(user_id):
    # conn = create_connection()
    # cursor = conn.cursor()
    secret = pyotp.random_base32()  # Generate a TOTP secret
    
    # st.write(g)
    # conn.commit()
    # conn.close()
    return secret
