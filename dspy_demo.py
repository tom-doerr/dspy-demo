import os
import dspy
from typing import Optional, List, Union, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time

console = Console()

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
        question = "What are the main benefits of using Stanford DSPy?"

        console.print(f"[yellow]Executing Q&A program with question: {question}[/yellow]")

        # Use DSPy's predictor directly with proper signature
        predictor = dspy.Predict("question -> answer")
        result = predictor(question=question).answer

        if not result:
            console.print("[red]Error: Failed to get valid response from Q&A program[/red]")
            return

        print_result(
            "Question & Answer Example",
            f"Q: {question}\nA: {result}"
        )
    except Exception as e:
        console.print(f"[red]Error in Q&A example: {str(e)}[/red]")

def run_summarization_example():
    """Run a text summarization example"""
    try:
        console.print("[yellow]Running summarization example...[/yellow]")
        text = """
        Stanford DSPy is a framework for solving complex language tasks by programming with foundation models.
        It provides a powerful yet simple interface for creating, composing, and optimizing language model programs.
        The framework enables developers to build reliable NLP applications while maintaining control over the development process.
        """

        console.print("[yellow]Executing summarization program...[/yellow]")

        # Use DSPy's predictor directly with proper signature
        predictor = dspy.Predict("text -> summary")
        result = predictor(text=text).summary

        if not result:
            console.print("[red]Error: Failed to get valid response from summarization program[/red]")
            return

        print_result(
            "Summarization Example",
            f"Original Text:\n{text}\n\nSummary:\n{result}"
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
        console.print("- Model: gemini/gemini-2.0-flash-lite-preview-02-05")
        console.print("- API Base: https://openrouter.ai/api/v1/chat/completions")

        lm = dspy.OpenRouterLM(
            model="gemini/gemini-2.0-flash-lite-preview-02-05",
            api_base="https://openrouter.ai/api/v1/chat/completions",
            api_key=api_key,
            headers={"HTTP-Referer": "https://replit.com", "X-Title": "Stanford DSPy Demo"},
            temperature=0.7
        )

        dspy.configure(lm=lm)
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