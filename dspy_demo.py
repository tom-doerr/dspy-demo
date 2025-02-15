import os
import dspy
from typing import Optional, List, Union, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
import logging
import litellm
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

def print_result(title: str, content: str):
    """Print a formatted result panel"""
    panel = Panel(
        Text(content),
        title=title,
        border_style="green"
    )
    console.print(panel)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def setup_dspy() -> bool:
    """Set up DSPy with OpenRouter configuration"""
    try:
        console.print("[yellow]Setting up DSPy...[/yellow]")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[red]OpenRouter API key not found[/red]")
            return False

        # Configure DSPy with OpenRouter model
        lm = dspy.LM(
            model="openai/gpt-3.5-turbo",
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            headers={
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Stanford DSPy Demo"
            },
            timeout=30
        )

        console.print("[yellow]Configuring DSPy...[/yellow]")
        dspy.configure(lm=lm)
        console.print("[green]Successfully configured DSPy[/green]")

        # Test the configuration with a simple completion
        logger.debug("Testing DSPy configuration...")
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="Is this a test?")
        logger.debug("Test response received: %s", result)

        return True
    except Exception as e:
        logger.error("DSPy setup failed: %s", str(e), exc_info=True)
        console.print(f"[red]Error setting up DSPy: {str(e)}[/red]")
        raise

def run_qa_example():
    """Run a question-answering example"""
    try:
        console.print("[yellow]Running Q&A example...[/yellow]")
        question = "What are the main benefits of using Stanford DSPy?"

        console.print(f"[yellow]Executing Q&A program with question: {question}[/yellow]")

        predictor = dspy.ChainOfThought("question -> answer")
        logger.debug("Created predictor instance")

        logger.debug("Sending request to OpenRouter...")
        result = predictor(question=question)
        logger.debug("Received response from predictor: %s", result)

        if not result or not hasattr(result, 'answer'):
            console.print("[red]Error: Failed to get valid response from Q&A program[/red]")
            return

        print_result(
            "Question & Answer Example",
            f"Q: {question}\nA: {result.answer}"
        )
    except Exception as e:
        logger.error("Detailed error in Q&A example: %s", str(e), exc_info=True)
        console.print(f"[red]Error in Q&A example: {str(e)}[/red]")

def main():
    """Main function to run the demonstration"""
    console.print("[bold green]Stanford DSPy Demo[/bold green]")
    console.print("=" * 80)
    console.print()

    if not setup_dspy():
        return

    # Add some delay between API calls to respect rate limits
    run_qa_example()
    time.sleep(2)

    console.print("[bold green]Demo completed![/bold green]")

if __name__ == "__main__":
    main()