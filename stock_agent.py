#!/usr/bin/env python3
"""Stock Quote Analysis Agent — an interactive CLI that uses Claude with tools
to look up stock quotes, retrieve price history, and provide analysis.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    pip install -r requirements.txt
    python stock_agent.py
"""

import json
import sys
from datetime import datetime, timedelta

import anthropic
import yfinance as yf

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_stock_quote(symbol: str) -> str:
    """Fetch the current quote for a stock ticker."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info or info.get("regularMarketPrice") is None:
        return json.dumps({"error": f"Could not find quote for '{symbol}'"})

    quote = {
        "symbol": info.get("symbol", symbol.upper()),
        "name": info.get("shortName") or info.get("longName", "N/A"),
        "price": info.get("regularMarketPrice"),
        "currency": info.get("currency", "USD"),
        "change": info.get("regularMarketChange"),
        "change_percent": info.get("regularMarketChangePercent"),
        "day_high": info.get("dayHigh"),
        "day_low": info.get("dayLow"),
        "volume": info.get("volume"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "dividend_yield": info.get("dividendYield"),
    }
    return json.dumps(quote)


def get_price_history(symbol: str, period: str = "1mo") -> str:
    """Fetch historical price data for a stock.

    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in valid_periods:
        return json.dumps({"error": f"Invalid period '{period}'. Use one of: {', '.join(sorted(valid_periods))}"})

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No history found for '{symbol}'"})

    records = []
    for date, row in hist.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        })

    summary = {
        "symbol": symbol.upper(),
        "period": period,
        "data_points": len(records),
        "start_date": records[0]["date"],
        "end_date": records[-1]["date"],
        "start_price": records[0]["close"],
        "end_price": records[-1]["close"],
        "price_change": round(records[-1]["close"] - records[0]["close"], 2),
        "price_change_pct": round(
            (records[-1]["close"] - records[0]["close"]) / records[0]["close"] * 100, 2
        ),
        "highest_close": max(r["close"] for r in records),
        "lowest_close": min(r["close"] for r in records),
        "avg_volume": int(sum(r["volume"] for r in records) / len(records)),
        "recent_prices": records[-10:],  # last 10 data points
    }
    return json.dumps(summary)


def compare_stocks(symbols: list[str]) -> str:
    """Compare key metrics across multiple stock tickers."""
    results = []
    for sym in symbols:
        ticker = yf.Ticker(sym)
        info = ticker.info
        if not info or info.get("regularMarketPrice") is None:
            results.append({"symbol": sym.upper(), "error": "Not found"})
            continue
        results.append({
            "symbol": info.get("symbol", sym.upper()),
            "name": info.get("shortName") or info.get("longName", "N/A"),
            "price": info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "change_percent": info.get("regularMarketChangePercent"),
        })
    return json.dumps(results)


# ---------------------------------------------------------------------------
# Tool definitions for the Claude API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_stock_quote",
        "description": (
            "Get the current real-time quote for a stock ticker symbol. "
            "Returns price, change, volume, market cap, P/E ratio, and more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT, GOOGL",
                }
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_price_history",
        "description": (
            "Get historical price data for a stock over a given period. "
            "Returns summary stats (price change, highs/lows, avg volume) "
            "and the most recent data points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, MSFT",
                },
                "period": {
                    "type": "string",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                    "description": "Time period for historical data. Defaults to 1mo.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "compare_stocks",
        "description": (
            "Compare key metrics (price, market cap, P/E, dividend yield, etc.) "
            "across multiple stock ticker symbols side by side."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols to compare, e.g. ['AAPL', 'MSFT', 'GOOGL']",
                }
            },
            "required": ["symbols"],
        },
    },
]

SYSTEM_PROMPT = """\
You are a knowledgeable stock market analysis assistant. You help users look up \
stock quotes, analyze price trends, and compare stocks.

Guidelines:
- Always use the available tools to fetch real data before answering.
- Present numbers clearly — format large numbers (e.g. $2.85T market cap) and \
  percentages (e.g. +1.23%).
- When analyzing trends, note key support/resistance levels, recent momentum, \
  and relevant context.
- If the user asks about a stock you don't recognize, try looking it up anyway — \
  the tool will return an error if it's invalid.
- Be concise but thorough. Offer follow-up suggestions like comparing with \
  peers or checking a different time period.
- Remind the user that this is informational only and not financial advice.
"""

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, tool_input: dict) -> str:
    """Route a tool call to the appropriate function."""
    if name == "get_stock_quote":
        return get_stock_quote(tool_input["symbol"])
    elif name == "get_price_history":
        return get_price_history(tool_input["symbol"], tool_input.get("period", "1mo"))
    elif name == "compare_stocks":
        return compare_stocks(tool_input["symbols"])
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Agentic chat loop
# ---------------------------------------------------------------------------

def run_agent():
    client = anthropic.Anthropic()
    messages: list[dict] = []

    print("=" * 60)
    print("  Stock Quote Analysis Agent")
    print("  Ask about any stock — quotes, history, comparisons.")
    print("  Type 'quit' or 'exit' to leave.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic loop — keep going until Claude stops calling tools
        while True:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                thinking={"type": "adaptive"},
                messages=messages,
            )

            # Collect assistant response
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Print text blocks from the final response
                for block in response.content:
                    if block.type == "text":
                        print(f"\nAssistant: {block.text}\n")
                break

            if response.stop_reason == "pause_turn":
                # Server-side tool hit iteration limit; re-send to continue
                continue

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [Calling {block.name}({json.dumps(block.input)})]")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls and not end_turn — shouldn't happen, but break
                for block in response.content:
                    if block.type == "text":
                        print(f"\nAssistant: {block.text}\n")
                break


if __name__ == "__main__":
    run_agent()
