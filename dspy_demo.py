import os
import dspy
from typing import Optional, List, Union, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time

console = Console()

class SimpleQA(dspy.Signature):
    """Simple question-answering signature"""
    input_question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the input question")

class Summarizer(dspy.Signature):
    """Text summarization signature"""
    input_text = dspy.InputField()
    summary = dspy.OutputField(desc="Concise summary of the input text")

def print_result(title: str, content: str):
    """Print formatted results using Rich"""
    panel = Panel(
        Text(content, style="blue"),
        title=title,
        border_style="green"
    )
    console.print(panel)
    console.print()

def run_qa_example():
    """Run a question-answering example"""
    try:
        console.print("[yellow]Running Q&A example...[/yellow]")
        qa_program = dspy.Predict(SimpleQA)
        question = "What are the main benefits of using Stanford DSPy?"

        console.print(f"[yellow]Executing Q&A program with question: {question}[/yellow]")
        result = qa_program(input_question=question)

        if not result or not hasattr(result, 'answer'):
            console.print("[red]Error: Failed to get valid response from Q&A program[/red]")
            return

        print_result(
            "Question & Answer Example",
            f"Q: {question}\nA: {result.answer}"
        )
    except Exception as e:
        console.print(f"[red]Error in Q&A example: {str(e)}[/red]")

def run_summarization_example():
    """Run a text summarization example"""
    try:
        console.print("[yellow]Running summarization example...[/yellow]")
        summarize_program = dspy.Predict(Summarizer)
        text = """
        Stanford DSPy is a framework for solving complex language tasks by programming with foundation models.
        It provides a powerful yet simple interface for creating, composing, and optimizing language model programs.
        The framework enables developers to build reliable NLP applications while maintaining control over the development process.
        """

        console.print("[yellow]Executing summarization program...[/yellow]")
        result = summarize_program(input_text=text)

        if not result or not hasattr(result, 'summary'):
            console.print("[red]Error: Failed to get valid response from summarization program[/red]")
            return

        print_result(
            "Summarization Example",
            f"Original Text:\n{text}\n\nSummary:\n{result.summary}"
        )
    except Exception as e:
        console.print(f"[red]Error in summarization example: {str(e)}[/red]")

def setup_dspy():
    """Configure DSPy with OpenRouter Gemini model"""
    try:
        console.print("[yellow]Setting up DSPy with OpenRouter Gemini model...[/yellow]")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

        # Configure OpenRouter endpoint
        console.print("[yellow]Configuring DSPy LM with OpenRouter settings:[/yellow]")
        console.print("- Model: google/gemini-pro")
        console.print("- API Base: https://openrouter.ai/api/v1/chat/completions")

        lm = dspy.LM(
            model_type="openai",  # Using OpenAI-compatible endpoint
            model="google/gemini-pro",  # Correct model identifier for OpenRouter
            api_base="https://openrouter.ai/api/v1/chat/completions",  # Specific endpoint for chat completions
            api_key=api_key,
            headers={
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Stanford DSPy Demo"
            },
            temperature=0.7,
            max_tokens=1000
        )

        dspy.settings.configure(lm=lm)
        console.print("[green]Successfully configured DSPy[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error setting up DSPy: {str(e)}[/red]")
        return False

def main():
    """Main function to run the demonstration"""
    console.print("[bold green]Stanford DSPy with OpenRouter Gemini Flash 2.0 Demo[/bold green]")
    console.print("=" * 80)
    console.print()

    if not setup_dspy():
        return

    # Add some delay between API calls to respect rate limits
    run_qa_example()
    time.sleep(2)

    run_summarization_example()
    time.sleep(2)

    console.print("[bold green]Demo completed![/bold green]")

if __name__ == "__main__":
    main()