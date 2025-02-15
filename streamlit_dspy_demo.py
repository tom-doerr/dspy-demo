import os
import streamlit as st
import dspy
import traceback

# Page config
st.set_page_config(page_title="DSPy Demo", page_icon="🤖")

def configure_dspy(model_name: str):
    """Configure DSPy with OpenRouter"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.error("Error: OPENROUTER_API_KEY environment variable is not set")
            return False

        # Configure DSPy with OpenRouter
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
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}\n\n{traceback.format_exc()}")
        return False

def run_dspy_demo():
    """Run a simple DSPy demo"""
    try:
        predictor = dspy.ChainOfThought(
            "question -> answer",
            instructions="You are a Stanford DSPy expert. Explain DSPy's features clearly and concisely."
        )
        question = "What is Stanford DSPy and what makes it different from traditional prompting?"
        result = predictor(question=question)
        st.write("Question:", question)
        st.write("Answer:", result.answer)
    except Exception as e:
        st.error(f"Error running demo: {str(e)}\n\n{traceback.format_exc()}")

# Main interface
st.title("DSPy Minimal Demo")

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

if st.button("Configure DSPy"):
    if configure_dspy(model_name):
        st.success(f"DSPy configured successfully with model: {model_name}")

if st.button("Run Demo"):
    run_dspy_demo()