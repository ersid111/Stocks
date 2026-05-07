from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()


def render_header(ticker: str, company: str, price: float, change_pct: float):
    direction = "▲" if change_pct and change_pct >= 0 else "▼"
    color = "green" if change_pct and change_pct >= 0 else "red"
    change_str = f"[{color}]{direction} {abs(change_pct):.2f}%[/{color}]" if change_pct else ""
    price_str = f"[bold white]${price:.2f}[/bold white]" if price else "[dim]N/A[/dim]"
    header = f"[bold cyan]INSTITUTIONAL ANALYSIS REPORT[/bold cyan]\n[bold yellow]{ticker}[/bold yellow] — {company or 'Unknown'}\n{price_str}  {change_str}"
    console.print(Panel(header, box=box.DOUBLE_EDGE, border_style="cyan"), justify="center")


def render_report(report_text: str, ticker: str, company: str = "", price: float = None, change_pct: float = None):
    console.print()
    render_header(ticker, company, price, change_pct)
    console.print()

    # Split on ## Section headers and render each as a panel
    sections = []
    current_title = None
    current_body = []

    for line in report_text.splitlines():
        if line.startswith("## Section"):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_body)))
            current_title = line.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_title is not None:
        sections.append((current_title, "\n".join(current_body)))

    if not sections:
        # No section headers found — render as a single block
        console.print(Markdown(report_text))
        return

    section_colors = [
        "cyan", "yellow", "green", "magenta", "blue", "red", "white"
    ]

    for i, (title, body) in enumerate(sections):
        color = section_colors[i % len(section_colors)]
        console.print(Rule(f"[bold {color}]{title}[/bold {color}]"))
        console.print(Markdown(body.strip()))
        console.print()

    console.print(Rule("[dim]End of Report[/dim]"))


def render_progress_message(message: str):
    console.print(f"[dim cyan]  → {message}[/dim cyan]")


def render_error(message: str):
    console.print(f"[bold red]✗ Error:[/bold red] {message}")


def render_success(message: str):
    console.print(f"[bold green]✓[/bold green] {message}")
