import os
import streamlit as st
import dspy
import traceback

# Page config
st.set_page_config(page_title="DSPy Demo", page_icon="🤖")

def configure_dspy():
    """Configure DSPy with OpenRouter"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.error("Error: OPENROUTER_API_KEY environment variable is not set")
            return False

        # Configure DSPy with OpenRouter
        lm = dspy.LM(
            model="openai/gpt-3.5-turbo",
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

if st.button("Configure DSPy"):
    if configure_dspy():
        st.success("DSPy configured successfully!")

if st.button("Run Demo"):
    run_dspy_demo()