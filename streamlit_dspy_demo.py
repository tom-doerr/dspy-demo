import streamlit as st

# Page config
st.set_page_config(page_title="DSPy Demo Test", page_icon="🤖")

# Main title
st.title("DSPy Tweet Generator")

# Sidebar
with st.sidebar:
    st.header("Model Configuration")
    st.write("Test configuration panel")

# Main content
st.write("Basic test of Streamlit functionality")