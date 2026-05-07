import os
from datetime import datetime


def save_report(report_text: str, ticker: str, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker.upper()}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    header = f"# Institutional Analysis Report: {ticker.upper()}\n\n"
    header += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return filepath
