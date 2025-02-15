import os
import dspy

def main():
    # Setup DSPy
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found")
        return
    
    # Configure DSPy with OpenRouter
    lm = dspy.LM(
        model="openai/gpt-3.5-turbo",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        headers={
            "HTTP-Referer": "https://replit.com",
            "X-Title": "Simple DSPy Demo"
        }
    )
    dspy.configure(lm=lm)
    
    # Create a simple predictor
    predictor = dspy.ChainOfThought("question -> answer")
    
    # Ask a question
    result = predictor(question="What is DSPy used for?")
    print(f"\nQuestion: What is DSPy used for?")
    print(f"Answer: {result.answer}")

if __name__ == "__main__":
    main()
