import os
import sys
import traceback
from typing import Optional, Callable
import streamlit as st
import dspy
from traceback_with_variables import activate_by_import

# Page config
st.set_page_config(
    page_title="Stanford DSPy Demos",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for storing error messages
if 'errors' not in st.session_state:
    st.session_state.errors = []

def configure_dspy() -> Optional[str]:
    """Configure DSPy with OpenRouter, returns error message if configuration fails"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Error: OPENROUTER_API_KEY environment variable is not set"
        
        # Configure DSPy with OpenRouter
        lm = dspy.LM(
            model="openai/gpt-3.5-turbo",
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            headers={
                "HTTP-Referer": "https://replit.com",
                "X-Title": "DSPy Demo Hub"
            }
        )
        dspy.configure(lm=lm)
        return None
    except Exception as e:
        return f"Error configuring DSPy: {str(e)}\n\n{traceback.format_exc()}"

def run_with_error_handling(func: Callable) -> None:
    """Run a function with proper error handling and display results in Streamlit"""
    try:
        with st.spinner('Running...'):
            result = func()
        st.success("Execution completed successfully!")
        st.write("Result:", result)
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        st.error(error_msg)
        st.session_state.errors.append(error_msg)

def simple_question_demo():
    """Run a simple DSPy question-answering demo"""
    predictor = dspy.ChainOfThought(
        "question -> answer",
        instructions="""You are a Stanford DSPy expert. DSPy is a framework for programming language models (LMs) 
        that allows developers to write Python code instead of using traditional prompts to create and optimize AI systems."""
    )
    question = "What is Stanford DSPy and how does it help developers work with language models?"
    result = predictor(question=question)
    return {
        "Question": question,
        "Answer": result.answer if hasattr(result, 'answer') else str(result)
    }

def main():
    st.title("Stanford DSPy Demo Hub")
    st.write("""
    This app demonstrates various capabilities of Stanford DSPy, a framework for programming language models.
    Each demo shows different aspects of DSPy's functionality with full error tracking.
    """)
    
    # Configuration section
    st.header("🔧 Configuration")
    if st.button("Configure DSPy"):
        error = configure_dspy()
        if error:
            st.error(error)
        else:
            st.success("DSPy configured successfully!")
    
    # Demo selection
    st.header("🚀 Available Demos")
    
    demo_options = {
        "Simple Q&A": simple_question_demo,
        # Add more demos here as needed
    }
    
    selected_demo = st.selectbox(
        "Select a demo to run:",
        options=list(demo_options.keys())
    )
    
    if st.button(f"Run {selected_demo}"):
        run_with_error_handling(demo_options[selected_demo])
    
    # Error Log Section
    st.header("📋 Error Log")
    if st.session_state.errors:
        if st.button("Clear Error Log"):
            st.session_state.errors = []
        
        for i, error in enumerate(st.session_state.errors):
            with st.expander(f"Error {i+1}"):
                st.code(error, language="python")
    else:
        st.info("No errors logged yet.")

if __name__ == "__main__":
    main()
