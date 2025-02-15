import streamlit as st
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log startup information
logger.info(f"Python version: {sys.version}")
logger.info("Starting Streamlit application...")

try:
    # Page config
    st.set_page_config(page_title="DSPy Demo Test", page_icon="🤖")
    logger.info("Page config set successfully")

    # Main title
    st.title("DSPy Tweet Generator")
    logger.info("Title rendered")

    # Sidebar
    with st.sidebar:
        st.header("Model Configuration")
        st.write("Test configuration panel")
    logger.info("Sidebar created")

    # Main content
    st.write("Basic test of Streamlit functionality")
    logger.info("Main content rendered")

except Exception as e:
    logger.error(f"Error in Streamlit app: {str(e)}")
    st.error(f"An error occurred: {str(e)}")