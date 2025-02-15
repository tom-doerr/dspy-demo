import os
import streamlit as st
import dspy
import traceback
from typing import Optional

# Page config
st.set_page_config(page_title="DSPy Demo", page_icon="🤖")
st.title("DSPy Demo with OpenRouter")

def configure_dspy(model_name: str) -> Optional[str]:
    """Configure DSPy with OpenRouter and return any error message"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Error: OPENROUTER_API_KEY environment variable is not set"

        lm = dspy.LM(
            model=model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            headers={
                "HTTP-Referer": "https://replit.com",
                "X-Title": "DSPy Demo"
            }
        )
        dspy.configure(lm=lm)
        return None
    except Exception as e:
        return f"Error configuring DSPy: {str(e)}"

def run_demo() -> tuple[str, str]:
    """Run the DSPy demo and return question and answer"""
    predictor = dspy.ChainOfThought(
        "question -> answer",
        instructions="You are a Stanford DSPy expert. Explain DSPy's features clearly and concisely."
    )
    question = "What is Stanford DSPy and what makes it different from traditional prompting?"
    result = predictor(question=question)
    return question, result.answer

# Model selection
models = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-sonnet",
]

# Add prefix option
add_prefix = st.checkbox("Add 'openrouter/' prefix", value=True)
selected_model = st.selectbox("Select Model", models)

# Update model name based on prefix choice
model_name = f"openrouter/{selected_model}" if add_prefix else selected_model

# Initialize session state
if 'dspy_configured' not in st.session_state:
    st.session_state.dspy_configured = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Configure DSPy when model changes
error_msg = configure_dspy(model_name)
if error_msg:
    st.error(error_msg)
    st.session_state.dspy_configured = False
    st.session_state.error_message = error_msg
else:
    st.success(f"DSPy configured successfully with model: {model_name}")
    st.session_state.dspy_configured = True
    st.session_state.error_message = None

# Show Run Demo button only after successful configuration
if st.session_state.dspy_configured:
    if st.button("Run Demo"):
        try:
            question, answer = run_demo()
            st.write("Question:", question)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error running demo: {str(e)}\n\n{traceback.format_exc()}")