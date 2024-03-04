import streamlit as st
from streamlit.runtime.state import SessionState

from pages import data_manipulation_page
from traitement.description import load_data

st.title("Mon Application Streamlit")
st.write("Bonjour depuis Streamlit!")

# Create a session state to store the data frame
state = SessionState.get(df=None)

# Load data if not loaded
if state.df is None:
    state.df = load_data()

# Render different pages based on user selection
page = st.sidebar.selectbox("Page", ["Data Manipulation"])

if page == "Data Manipulation":
    data_manipulation_page.show(state)