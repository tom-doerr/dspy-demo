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

# Configure DSPy with our language model
dspy.configure(lm=lm)

# Create a ChainOfThought predictor with system context
predictor = dspy.ChainOfThought(
    "question -> answer",
    instructions="""You are a Stanford DSPy expert. DSPy is a framework for programming language models (LMs) 
    that allows developers to write Python code instead of using traditional prompts to create and optimize AI systems."""
)

# Ask about DSPy's purpose
question = "What is Stanford DSPy's main purpose and how does it differ from traditional prompting?"
result = predictor(question=question)
print(f"\nQuestion: {question}")
print(f"Answer: {result.answer}")