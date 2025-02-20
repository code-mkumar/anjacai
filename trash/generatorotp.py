import pyotp
import time

# Define your secret key (keep it safe)
secret_key = input("enter the otp")  # Replace with your actual secret

# Generate TOTP valid for 30 seconds
while True:
    totp = pyotp.TOTP(secret_key, interval=30)
    otp = totp.now()

    print("Your OTP:", otp)
    time.sleep(30)
