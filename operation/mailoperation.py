import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, sender_password, recipient_email, subject, body):
    try:
        # Set up the email server (Gmail SMTP server in this case)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        # Create a secure connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()

        # Log in to the email account
        server.login(sender_email, sender_password)

        # Create the email
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject

        # Add the body to the email
        message.attach(MIMEText(body, "plain"))

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

# Example usage
# if __name__ == "__main__":
def email(user,email):
    sender_email = "sopnan500@gmail.com"
    sender_password = "salkzgccfjqykude"
    recipient_email = email
    subject = "Thank You for Your Feedback on AnjacAI!"
    body = f"""
            Dear {user},

            Thank you for taking the time to share your valuable feedback on AnjacAI. We truly appreciate your thoughts and suggestions, as they help us improve and provide a better experience for you.

            If you have any further ideas or questions, feel free to reach out to us.

            Best regards,  
            The AnjacAI Team

        """

    send_email(sender_email, sender_password, recipient_email, subject, body)


# EMAIL_USER='sopnan500@gmail.com'
# EMAIL_PASS='salkzgccfjqykude'