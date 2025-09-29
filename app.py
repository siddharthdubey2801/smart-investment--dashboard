import streamlit as st
import firebase_admin
import plotly.express as px
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Assuming this is how it's imported
import nltk
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd # Assuming you use pandas for data handling

# ----------------------------------------------------
# --- ROBUST FIREBASE INITIALIZATION ---
# This block ensures Firebase is initialized only once per session.
# It resolves the "default Firebase app already exists" error.
# ----------------------------------------------------

# 1. Use st.session_state to track initialization status
if "firebase_initialized" not in st.session_state:
    st.session_state["firebase_initialized"] = False

if not st.session_state["firebase_initialized"]:
    
    # 2. Use the internal check AND the session state check
    if not firebase_admin._apps:
        try:
            # 3. Get the credentials dictionary from Streamlit Secrets
            cred_dict = st.secrets["firebase"] 
            
            # Initialize the app with the valid credentials
            cred = firebase_admin.credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            
            # Set the flag to True after successful initialization
            st.session_state["firebase_initialized"] = True
            
        except Exception as e:
            # Display a detailed error for debugging, but prevent the app from crashing entirely
            st.error("Failed to initialize Firebase credentials. Please check your .streamlit/secrets.toml file.")
            st.code(f"Error details: {e}")
            # Exit the app if Firebase is critical
            st.stop()


# ----------------------------------------------------
# --- START OF YOUR MAIN STREAMLIT APPLICATION LOGIC ---
# ----------------------------------------------------

# 4. Your Application's Authentication and Dashboard Logic

# Placeholder for your actual login function (assuming it returns True on success)
def check_login(email, password):
    # YOUR ACTUAL FIREBASE AUTHENTICATION LOGIC GOES HERE
    # Example:
    # user = auth.get_user_by_email(email)
    # if auth.verify_password(password, user.password_hash):
    #     return True
    
    # For now, we'll assume a dummy check since the initialization is fixed
    if email == "ldharthzay2341@gmail.com" and password == "siddharth@123":
         return True
    return False

# Display the content based on login state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    # --- DASHBOARD CONTENT GOES HERE ---
    st.title("✅ Smart Investment Dashboard")
    st.write("Welcome, you are successfully logged in!")
    # Example of using your imported modules:
    # st.plotly_chart(px.line(data_frame=df, x='Date', y='Value'))
    
else:
    st.title("Please log in to continue")
    
    with st.form("login_form"):
        st.write("Login")
        email = st.text_input("Email", value="ldharthzay2341@gmail.com")
        password = st.text_input("Password", type="password", value="siddharth@123")
        submitted = st.form_submit_button("Login")

        if submitted:
            # Replace this with your actual Firebase authentication call
            if check_login(email, password):
                st.session_state['logged_in'] = True
                st.experimental_rerun()
            else:
                st.error("Login failed: Invalid email or password.")