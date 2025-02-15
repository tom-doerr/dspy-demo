import os
import dspy
from typing import Optional, List, Union, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
import litellm
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        logger.debug("Created predictor instance")

        logger.debug("Sending request to OpenRouter...")
        result = predictor(question=question).answer
        logger.debug(f"Received response: {result}")

        if not result:
            console.print("[red]Error: Failed to get valid response from Q&A program[/red]")
            return

        print_result(
            "Question & Answer Example",
            f"Q: {question}\nA: {result}"
        )
    except Exception as e:
        logger.error(f"Detailed error in Q&A example: {str(e)}", exc_info=True)
        console.print(f"[red]Error in Q&A example: {str(e)}[/red]")
        console.print(f"[red]Error type: {type(e)}[/red]")

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
        logger.debug("Created summarization predictor")

        logger.debug("Sending request to OpenRouter...")
        result = predictor(text=text).summary
        logger.debug(f"Received response: {result}")

        if not result:
            console.print("[red]Error: Failed to get valid response from summarization program[/red]")
            return

        print_result(
            "Summarization Example",
            f"Original Text:\n{text}\n\nSummary:\n{result}"
        )
    except Exception as e:
        logger.error(f"Detailed error in summarization example: {str(e)}", exc_info=True)
        console.print(f"[red]Error in summarization example: {str(e)}[/red]")
        console.print(f"[red]Error type: {type(e)}[/red]")

def test_api_connection():
    """Test the OpenRouter API connection"""
    try:
        console.print("[yellow]Testing OpenRouter API connection...[/yellow]")
        api_key = os.getenv("OPENROUTER_API_KEY")

        # Make a simple test request using litellm directly
        logger.debug("Making test request to OpenRouter")
        response = litellm.completion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://openrouter.ai/api/v1/chat/completions",
            api_key=api_key,
            headers={
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Stanford DSPy Demo"
            }
        )
        logger.debug(f"Test response received: {response}")
        return True
    except Exception as e:
        logger.error(f"API test failed: {str(e)}", exc_info=True)
        console.print(f"[red]API test failed: {str(e)}[/red]")
        return False

def setup_dspy():
    """Configure DSPy with OpenRouter model"""
    try:
        console.print("[yellow]Setting up DSPy with OpenRouter model...[/yellow]")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

        # Test API connection first
        if not test_api_connection():
            raise ValueError("Failed to connect to OpenRouter API")

        console.print("[yellow]Configuring DSPy LM with OpenRouter settings...[/yellow]")

        # Configure OpenRouter endpoint with correct model identifier
        lm = dspy.LM(
            model="openai/gpt-3.5-turbo",  # Using a well-supported model
            api_base="https://openrouter.ai/api/v1/chat/completions",
            api_key=api_key,
            headers={
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Stanford DSPy Demo"
            },
            model_type="chat"  # Using chat completion type
        )

        console.print("[yellow]Created OpenRouter LM instance, configuring DSPy...[/yellow]")
        dspy.configure(lm=lm)
        console.print("[green]Successfully configured DSPy[/green]")
        return True
    except Exception as e:
        logger.error(f"Error setting up DSPy: {str(e)}", exc_info=True)
        console.print(f"[red]Error setting up DSPy: {str(e)}[/red]")
        console.print(f"[red]Error type: {type(e)}[/red]")
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