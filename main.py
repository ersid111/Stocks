#!/usr/bin/env python3
import argparse
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_pipeline(ticker: str, model: str, output_dir: str):
    ticker = ticker.upper().strip()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:

        # ── Data fetching ──────────────────────────────────────────────────
        task = progress.add_task(f"Fetching market data for {ticker}...", total=None)
        from data.market_data import fetch_market_snapshot
        snap = fetch_market_snapshot(ticker)
        if snap.current_price is None:
            progress.stop()
            console.print(f"[bold red]✗[/bold red] Could not fetch price data for {ticker}. Check the ticker symbol.")
            sys.exit(1)
        progress.update(task, description=f"[green]✓[/green] Market data fetched — ${snap.current_price:.2f}")

        task2 = progress.add_task("Fetching options chain...", total=None)
        from data.options_data import fetch_options_chain
        chain = fetch_options_chain(ticker, snap.current_price)
        if chain.error:
            progress.update(task2, description=f"[yellow]⚠[/yellow] Options: {chain.error}")
        else:
            progress.update(task2, description=f"[green]✓[/green] Options chain fetched ({len(chain.expiries)} expiries)")

        task3 = progress.add_task("Fetching macro & news data...", total=None)
        from data.macro_data import fetch_macro_context, fetch_news_sentiment
        macro = fetch_macro_context()
        news = fetch_news_sentiment(ticker)
        macro_status = "unavailable" if macro.error else f"VIX={macro.vix}"
        progress.update(task3, description=f"[green]✓[/green] Macro & news ({macro_status})")

        # ── Analysis ───────────────────────────────────────────────────────
        task4 = progress.add_task("Computing technical indicators...", total=None)
        from analysis.technical import compute_technical_indicators
        ind = compute_technical_indicators(snap.ohlcv)
        progress.update(task4, description=f"[green]✓[/green] Technicals — RSI={ind.rsi:.1f if ind.rsi else 'N/A'}, ADX={ind.adx:.1f if ind.adx else 'N/A'}")

        task5 = progress.add_task("Analyzing market structure...", total=None)
        from analysis.market_structure import (
            find_support_resistance, classify_trend,
            compute_volume_profile, detect_institutional_accumulation
        )
        sr_levels = find_support_resistance(snap.ohlcv)
        trend = classify_trend(ind, snap.current_price)
        vp = compute_volume_profile(snap.ohlcv)
        accum = detect_institutional_accumulation(ind, snap.ohlcv)
        progress.update(task5, description=f"[green]✓[/green] Structure — {trend.ema_alignment} EMA, {len(sr_levels)} S/R levels")

        task6 = progress.add_task("Computing options analytics...", total=None)
        from analysis.options_math import (
            calculate_max_pain, calculate_gamma_exposure,
            calculate_iv_skew, calculate_expected_move, compute_put_call_metrics
        )
        max_pain = calculate_max_pain(chain.calls, chain.puts)
        gex = calculate_gamma_exposure(chain.calls, chain.puts, snap.current_price)
        iv_skew = calculate_iv_skew(chain.calls, chain.puts, snap.current_price)

        nearest_dte = 30
        if chain.expiries:
            from datetime import datetime
            try:
                nearest_dte = max((datetime.strptime(chain.expiries[0], "%Y-%m-%d") - datetime.now()).days, 1)
            except Exception:
                pass

        expected_move = calculate_expected_move(snap.current_price, iv_skew.atm_iv, nearest_dte, chain.calls, chain.puts)
        pc = compute_put_call_metrics(chain.calls, chain.puts)
        gex_str = f"GEX {gex.regime}, flip@{gex.gamma_flip}" if gex.gamma_flip else f"GEX {gex.regime}"
        progress.update(task6, description=f"[green]✓[/green] Options — MaxPain={max_pain.max_pain}, {gex_str}")

        task7 = progress.add_task("Building scenario model...", total=None)
        from analysis.scenarios import build_scenarios
        scenarios = build_scenarios(ind, trend, pc, sr_levels, vp, snap.current_price, snap.ohlcv)
        bull_p = f"{scenarios.bull.probability:.0%}" if scenarios.bull else "N/A"
        bear_p = f"{scenarios.bear.probability:.0%}" if scenarios.bear else "N/A"
        progress.update(task7, description=f"[green]✓[/green] Scenarios — Bull {bull_p} / Bear {bear_p}")

        # ── AI Analysis ────────────────────────────────────────────────────
        task8 = progress.add_task(f"Building data payload...", total=None)
        from ai.prompt_builder import build_analysis_payload
        payload = build_analysis_payload(
            snap=snap, ind=ind, trend=trend, sr_levels=sr_levels,
            vp=vp, accum=accum, chain=chain, max_pain=max_pain,
            gex=gex, iv_skew=iv_skew, expected_move=expected_move,
            pc=pc, scenarios=scenarios, macro=macro, news=news,
        )
        progress.update(task8, description=f"[green]✓[/green] Payload built ({len(payload):,} chars)")

        task9 = progress.add_task(f"Generating institutional report via Claude ({model})...", total=None)
        from ai.claude_client import generate_analysis
        report_text, cache_info = generate_analysis(payload, model=model)
        progress.update(task9, description=f"[green]✓[/green] Report generated{cache_info}")

    # ── Output ─────────────────────────────────────────────────────────────
    from reporting.terminal import render_report
    from reporting.markdown_writer import save_report

    render_report(
        report_text=report_text,
        ticker=ticker,
        company=snap.company_name or "",
        price=snap.current_price,
        change_pct=snap.day_change_pct,
    )

    saved_path = save_report(report_text, ticker, output_dir)
    console.print(f"\n[bold green]Report saved →[/bold green] {saved_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Institutional-grade stock & options analysis powered by Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python main.py --ticker AAPL\n  python main.py --ticker SPY --model claude-opus-4-7\n  python main.py --ticker TSLA --output-dir ./reports",
    )
    parser.add_argument("--ticker", "-t", required=True, help="Stock ticker symbol (e.g. AAPL, SPY, TSLA)")
    parser.add_argument("--model", "-m", default="claude-opus-4-7", help="Claude model ID (default: claude-opus-4-7)")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Directory for saved reports (default: outputs)")

    args = parser.parse_args()

    if not __import__("config").ANTHROPIC_API_KEY:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    try:
        run_pipeline(ticker=args.ticker, model=args.model, output_dir=args.output_dir)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
