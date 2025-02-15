import os
import dspy

# Get OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found")
    exit(1)

# Configure DSPy with OpenRouter
lm = dspy.LM(
    model="openai/gpt-3.5-turbo",
    api_base="https://openrouter.ai/api/v1",
    api_key=api_key,
    headers={
        "HTTP-Referer": "https://replit.com",
        "X-Title": "Super Simple DSPy Demo"
    }
)
dspy.configure(lm=lm)

# Create a basic predictor and ask a question
predictor = dspy.ChainOfThought("question -> answer")
result = predictor(question="What is DSPy and what is it used for?")
print(f"\nQuestion: What is DSPy and what is it used for?")
print(f"Answer: {result.answer}")
