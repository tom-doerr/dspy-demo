import os
import dspy
from rich import print

def setup_dspy():
    """Configure DSPy with OpenRouter"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[red]Error: OPENROUTER_API_KEY environment variable is not set[/red]")
        return False
    
    # Configure DSPy with OpenRouter
    lm = dspy.LM(
        model="openai/gpt-3.5-turbo",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        headers={
            "HTTP-Referer": "https://replit.com",
            "X-Title": "Minimal DSPy Demo"
        }
    )
    dspy.configure(lm=lm)
    return True

def ask_question(question: str):
    """Ask a question using DSPy ChainOfThought"""
    try:
        # Create a predictor that uses chain-of-thought reasoning
        predictor = dspy.ChainOfThought("question -> answer")
        
        # Get the response
        result = predictor(question=question)
        
        print(f"\nQ: {question}")
        print(f"A: {result.answer}\n")
        
    except Exception as e:
        print(f"[red]Error getting response: {str(e)}[/red]")

def main():
    print("[bold green]Minimal DSPy Demo[/bold green]")
    
    if not setup_dspy():
        return
    
    # Example question
    question = "What are three main benefits of using DSPy for AI development?"
    ask_question(question)

if __name__ == "__main__":
    main()
