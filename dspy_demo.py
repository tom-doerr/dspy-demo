import os
import dspy
from typing import Optional, List, Union, Dict, Any
import openrouter
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
import json
import httpx

console = Console()

class OpenRouterGeminiLM(dspy.LM):
    """Custom Language Model class for OpenRouter Gemini Flash 2.0"""

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")

        self.api_key = api_key
        self.api_base = "https://openrouter.ai/api/v1/chat/completions"
        # Initialize kwargs for DSPy compatibility
        self.kwargs = {
            'model': 'google/gemini-pro',
            'temperature': 0.7,
            'max_tokens': 1000
        }

    def basic_request(self, prompt: str, **kwargs) -> Optional[str]:
        """Send a basic completion request to OpenRouter API"""
        try:
            console.print(f"[yellow]Sending request to OpenRouter API...[/yellow]")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://replit.com",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.kwargs['model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get('temperature', self.kwargs['temperature']),
                "max_tokens": kwargs.get('max_tokens', self.kwargs['max_tokens'])
            }

            console.print(f"[yellow]Request payload: {json.dumps(payload, indent=2)}[/yellow]")

            with httpx.Client() as client:
                response = client.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )

                console.print(f"[yellow]Response status code: {response.status_code}[/yellow]")

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    console.print(f"[green]Successfully received response[/green]")
                    return content
                else:
                    console.print(f"[red]Error: API returned status code {response.status_code}[/red]")
                    console.print(f"[red]Response: {response.text}[/red]")
                    return None

        except Exception as e:
            console.print(f"[red]Error in API request: {str(e)}[/red]")
            return None

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion for a prompt"""
        # Merge instance kwargs with method kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        result = self.basic_request(prompt, **merged_kwargs)
        if result is None:
            raise ValueError("Failed to generate response from OpenRouter API")
        return result

    def __call__(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Make the LM callable for DSPy compatibility"""
        if isinstance(prompts, list):
            return [self.generate(p, **kwargs) for p in prompts]
        return self.generate(prompts, **kwargs)

class SimpleQA(dspy.Signature):
    """Simple question-answering signature"""
    input_question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the input question")

    def forward(self, input_question: str) -> Dict[str, Any]:
        """Forward pass for Q&A"""
        response = self.lm(input_question)
        return {'answer': response}

class Summarizer(dspy.Signature):
    """Text summarization signature"""
    input_text = dspy.InputField()
    summary = dspy.OutputField(desc="Concise summary of the input text")

    def forward(self, input_text: str) -> Dict[str, Any]:
        """Forward pass for summarization"""
        prompt = f"Please summarize the following text:\n{input_text}"
        response = self.lm(prompt)
        return {'summary': response}

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
        lm = OpenRouterGeminiLM()
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