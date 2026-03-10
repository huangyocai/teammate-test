#!/usr/bin/env python3
"""Stock Quote Analysis Agent — an interactive CLI powered by Claude with
skills (activatable tool bundles) and MCP server support.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    pip install -r requirements.txt
    python stock_agent.py
    python stock_agent.py --skills technical,sector
    python stock_agent.py --mcp-server filesystem '{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import anthropic
import yfinance as yf
from anthropic import beta_async_tool

# ── MCP imports (optional — graceful if mcp not installed) ──────────────
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from anthropic.lib.tools.mcp import async_mcp_tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ===========================================================================
# Core stock tools (always available)
# ===========================================================================

@beta_async_tool
async def get_stock_quote(symbol: str) -> str:
    """Get the current real-time quote for a stock ticker symbol.
    Returns price, change, volume, market cap, P/E ratio, and more.

    Args:
        symbol: Stock ticker symbol, e.g. AAPL, MSFT, GOOGL.
    """
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


@beta_async_tool
async def get_price_history(symbol: str, period: str = "1mo") -> str:
    """Get historical price data for a stock over a given period.
    Returns summary stats (price change, highs/lows, avg volume) and recent data points.

    Args:
        symbol: Stock ticker symbol, e.g. AAPL, MSFT.
        period: Time period — one of 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    """
    valid = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in valid:
        return json.dumps({"error": f"Invalid period '{period}'. Use one of: {', '.join(sorted(valid))}"})

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No history found for '{symbol}'"})

    records = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        }
        for date, row in hist.iterrows()
    ]

    return json.dumps({
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
        "recent_prices": records[-10:],
    })


@beta_async_tool
async def compare_stocks(symbols: list[str]) -> str:
    """Compare key metrics (price, market cap, P/E, dividend yield, etc.)
    across multiple stock ticker symbols side by side.

    Args:
        symbols: List of stock ticker symbols to compare, e.g. ["AAPL", "MSFT"].
    """
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


CORE_TOOLS = [get_stock_quote, get_price_history, compare_stocks]


# ===========================================================================
# Skills — activatable tool + prompt bundles
# ===========================================================================

@dataclass
class Skill:
    """A named bundle of extra tools and system-prompt additions."""
    name: str
    description: str
    tools: list  # list of @beta_async_tool-decorated functions
    prompt_addition: str


# ── Technical Analysis skill ────────────────────────────────────────────

@beta_async_tool
async def compute_moving_averages(symbol: str, period: str = "6mo") -> str:
    """Compute simple moving averages (SMA-20, SMA-50, SMA-200) for a stock.

    Args:
        symbol: Stock ticker symbol.
        period: History period to use for calculation (default 6mo).
    """
    hist = yf.Ticker(symbol).history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No data for '{symbol}'"})

    closes = list(hist["Close"])
    result = {"symbol": symbol.upper(), "current_price": round(closes[-1], 2)}
    for window in (20, 50, 200):
        if len(closes) >= window:
            sma = round(statistics.mean(closes[-window:]), 2)
            result[f"sma_{window}"] = sma
            result[f"price_vs_sma_{window}"] = "above" if closes[-1] > sma else "below"
        else:
            result[f"sma_{window}"] = None
    return json.dumps(result)


@beta_async_tool
async def compute_rsi(symbol: str, period: str = "3mo", rsi_period: int = 14) -> str:
    """Compute the Relative Strength Index (RSI) for a stock.

    Args:
        symbol: Stock ticker symbol.
        period: History period to fetch (default 3mo).
        rsi_period: Number of periods for RSI calculation (default 14).
    """
    hist = yf.Ticker(symbol).history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No data for '{symbol}'"})

    closes = list(hist["Close"])
    if len(closes) < rsi_period + 1:
        return json.dumps({"error": "Not enough data points for RSI"})

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = statistics.mean(gains[:rsi_period])
    avg_loss = statistics.mean(losses[:rsi_period])

    for i in range(rsi_period, len(deltas)):
        avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
        avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = round(100 - (100 / (1 + rs)), 2)

    signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    return json.dumps({
        "symbol": symbol.upper(),
        "rsi": rsi,
        "signal": signal,
        "period_used": period,
        "rsi_period": rsi_period,
    })


@beta_async_tool
async def compute_volatility(symbol: str, period: str = "3mo") -> str:
    """Compute daily return volatility (std dev) and average daily range for a stock.

    Args:
        symbol: Stock ticker symbol.
        period: History period (default 3mo).
    """
    hist = yf.Ticker(symbol).history(period=period)
    if hist.empty:
        return json.dumps({"error": f"No data for '{symbol}'"})

    closes = list(hist["Close"])
    highs = list(hist["High"])
    lows = list(hist["Low"])
    daily_returns = [(closes[i] - closes[i - 1]) / closes[i - 1] * 100
                     for i in range(1, len(closes))]
    daily_ranges = [(h - l) / l * 100 for h, l in zip(highs, lows)]

    return json.dumps({
        "symbol": symbol.upper(),
        "daily_return_std_pct": round(statistics.stdev(daily_returns), 4),
        "avg_daily_range_pct": round(statistics.mean(daily_ranges), 4),
        "max_daily_return_pct": round(max(daily_returns), 4),
        "min_daily_return_pct": round(min(daily_returns), 4),
        "data_points": len(daily_returns),
    })


SKILL_TECHNICAL = Skill(
    name="technical",
    description="Technical analysis indicators — SMA, RSI, volatility",
    tools=[compute_moving_averages, compute_rsi, compute_volatility],
    prompt_addition=(
        "\n\nYou have technical analysis skills active. When discussing stock "
        "trends, compute and reference moving averages (SMA-20/50/200), RSI, "
        "and volatility metrics to give data-driven technical assessments."
    ),
)


# ── Sector Analysis skill ──────────────────────────────────────────────

@beta_async_tool
async def get_sector_performance(symbol: str) -> str:
    """Get the sector and industry for a stock plus its key sector peers.

    Args:
        symbol: Stock ticker symbol.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info:
        return json.dumps({"error": f"No info for '{symbol}'"})

    return json.dumps({
        "symbol": info.get("symbol", symbol.upper()),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "full_time_employees": info.get("fullTimeEmployees"),
        "recommendation": info.get("recommendationKey"),
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
    })


@beta_async_tool
async def get_financials_summary(symbol: str) -> str:
    """Get key financial metrics — revenue, earnings, margins, debt ratios.

    Args:
        symbol: Stock ticker symbol.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info:
        return json.dumps({"error": f"No info for '{symbol}'"})

    return json.dumps({
        "symbol": info.get("symbol", symbol.upper()),
        "revenue": info.get("totalRevenue"),
        "gross_margins": info.get("grossMargins"),
        "operating_margins": info.get("operatingMargins"),
        "profit_margins": info.get("profitMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "return_on_assets": info.get("returnOnAssets"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "free_cash_flow": info.get("freeCashflow"),
    })


SKILL_SECTOR = Skill(
    name="sector",
    description="Sector & fundamental analysis — financials, analyst targets, peers",
    tools=[get_sector_performance, get_financials_summary],
    prompt_addition=(
        "\n\nYou have sector/fundamental analysis skills active. When analyzing "
        "stocks, fetch sector context, analyst targets, and financial metrics "
        "to provide a well-rounded fundamental assessment."
    ),
)


# ── Portfolio skill ────────────────────────────────────────────────────

_portfolio: dict[str, float] = {}  # symbol -> shares (in-memory)


@beta_async_tool
async def portfolio_add(symbol: str, shares: float) -> str:
    """Add shares of a stock to the tracked portfolio.

    Args:
        symbol: Stock ticker symbol.
        shares: Number of shares to add (can be fractional).
    """
    symbol = symbol.upper()
    _portfolio[symbol] = _portfolio.get(symbol, 0) + shares
    return json.dumps({"message": f"Added {shares} shares of {symbol}. Total: {_portfolio[symbol]}"})


@beta_async_tool
async def portfolio_remove(symbol: str, shares: float) -> str:
    """Remove shares of a stock from the tracked portfolio.

    Args:
        symbol: Stock ticker symbol.
        shares: Number of shares to remove.
    """
    symbol = symbol.upper()
    if symbol not in _portfolio:
        return json.dumps({"error": f"{symbol} not in portfolio"})
    _portfolio[symbol] = max(0, _portfolio[symbol] - shares)
    if _portfolio[symbol] == 0:
        del _portfolio[symbol]
        return json.dumps({"message": f"Removed all {symbol} from portfolio"})
    return json.dumps({"message": f"Removed {shares} shares of {symbol}. Remaining: {_portfolio[symbol]}"})


@beta_async_tool
async def portfolio_view() -> str:
    """View current portfolio holdings with live market values."""
    if not _portfolio:
        return json.dumps({"message": "Portfolio is empty"})

    holdings = []
    total_value = 0.0
    for sym, shares in sorted(_portfolio.items()):
        ticker = yf.Ticker(sym)
        price = ticker.info.get("regularMarketPrice", 0) or 0
        value = price * shares
        total_value += value
        holdings.append({
            "symbol": sym,
            "shares": shares,
            "price": price,
            "market_value": round(value, 2),
        })

    return json.dumps({
        "holdings": holdings,
        "total_market_value": round(total_value, 2),
        "number_of_positions": len(holdings),
    })


SKILL_PORTFOLIO = Skill(
    name="portfolio",
    description="Track a portfolio — add/remove positions, view holdings with live values",
    tools=[portfolio_add, portfolio_remove, portfolio_view],
    prompt_addition=(
        "\n\nYou have portfolio tracking skills active. You can help the user "
        "manage a virtual portfolio — add positions, remove positions, and view "
        "their current holdings with live market values."
    ),
)


# ── Skill registry ─────────────────────────────────────────────────────

ALL_SKILLS: dict[str, Skill] = {
    s.name: s for s in [SKILL_TECHNICAL, SKILL_SECTOR, SKILL_PORTFOLIO]
}


# ===========================================================================
# MCP server connection
# ===========================================================================

@dataclass
class MCPConnection:
    """Holds an active MCP session plus the tools converted for the Anthropic API."""
    name: str
    session: ClientSession
    tools: list  # converted via async_mcp_tool


async def connect_mcp_server(
    name: str,
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    exit_stack: AsyncExitStack | None = None,
) -> MCPConnection:
    """Spawn an MCP stdio server and return its tools ready for the tool runner."""
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "MCP support requires the 'mcp' package. "
            "Install with: pip install 'anthropic[mcp]' mcp"
        )

    params = StdioServerParameters(command=command, args=args, env=env)
    read, write = await (exit_stack or AsyncExitStack()).enter_async_context(
        stdio_client(params)
    )
    session = await (exit_stack or AsyncExitStack()).enter_async_context(
        ClientSession(read, write)
    )
    await session.initialize()

    tools_result = await session.list_tools()
    converted = [async_mcp_tool(t, session) for t in tools_result.tools]
    return MCPConnection(name=name, session=session, tools=converted)


# ===========================================================================
# System prompt builder
# ===========================================================================

BASE_SYSTEM_PROMPT = """\
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


def build_system_prompt(
    active_skills: list[Skill],
    mcp_connections: list[MCPConnection],
) -> str:
    prompt = BASE_SYSTEM_PROMPT
    if active_skills:
        prompt += "\n\n--- Active Skills ---"
        for skill in active_skills:
            prompt += skill.prompt_addition
    if mcp_connections:
        prompt += "\n\n--- Connected MCP Servers ---"
        for conn in mcp_connections:
            tool_names = [t.name for t in conn.tools]
            prompt += f"\n- {conn.name}: tools = {tool_names}"
        prompt += (
            "\nYou can use MCP server tools just like any other tool. "
            "They have been automatically discovered and are available to you."
        )
    return prompt


# ===========================================================================
# Agentic chat loop (async, tool runner)
# ===========================================================================

async def run_agent(
    active_skill_names: list[str],
    mcp_server_specs: list[tuple[str, dict]],
) -> None:
    client = anthropic.AsyncAnthropic()

    # ── Resolve skills ──────────────────────────────────────────────────
    active_skills: list[Skill] = []
    for name in active_skill_names:
        if name in ALL_SKILLS:
            active_skills.append(ALL_SKILLS[name])
        else:
            print(f"Warning: unknown skill '{name}', skipping.")

    # ── Build tool list ─────────────────────────────────────────────────
    all_tools = list(CORE_TOOLS)
    for skill in active_skills:
        all_tools.extend(skill.tools)

    # ── Connect MCP servers ─────────────────────────────────────────────
    mcp_connections: list[MCPConnection] = []
    exit_stack = AsyncExitStack()
    async with exit_stack:
        for server_name, spec in mcp_server_specs:
            try:
                conn = await connect_mcp_server(
                    name=server_name,
                    command=spec["command"],
                    args=spec.get("args", []),
                    env=spec.get("env"),
                    exit_stack=exit_stack,
                )
                mcp_connections.append(conn)
                all_tools.extend(conn.tools)
                print(f"  MCP [{server_name}]: connected — {len(conn.tools)} tool(s)")
            except Exception as e:
                print(f"  MCP [{server_name}]: failed to connect — {e}")

        system_prompt = build_system_prompt(active_skills, mcp_connections)

        # ── Banner ──────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  Stock Quote Analysis Agent")
        print("  Ask about any stock — quotes, history, comparisons.")
        if active_skills:
            print(f"  Skills: {', '.join(s.name for s in active_skills)}")
        if mcp_connections:
            print(f"  MCP: {', '.join(c.name for c in mcp_connections)}")
        print("  Type 'quit' or 'exit' to leave.")
        print("=" * 60)
        print()

        messages: list[dict] = []

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ").strip()
                )
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            # Handle runtime skill toggling
            if user_input.lower().startswith("/skill"):
                _handle_skill_command(user_input, active_skills, all_tools)
                system_prompt = build_system_prompt(active_skills, mcp_connections)
                continue

            if user_input.lower() == "/tools":
                _handle_tools_command(all_tools, active_skills, mcp_connections)
                continue

            messages.append({"role": "user", "content": user_input})

            # Agentic loop
            while True:
                response = await client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    system=system_prompt,
                    tools=all_tools,
                    thinking={"type": "adaptive"},
                    messages=messages,
                )

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if block.type == "text":
                            print(f"\nAssistant: {block.text}\n")
                    break

                if response.stop_reason == "pause_turn":
                    continue

                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [Calling {block.name}({json.dumps(block.input)})]")
                        result = await _execute_tool_async(block.name, block.input, all_tools)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    for block in response.content:
                        if block.type == "text":
                            print(f"\nAssistant: {block.text}\n")
                    break


async def _execute_tool_async(name: str, tool_input: dict, all_tools: list) -> str:
    """Dispatch a tool call to the matching @beta_async_tool function."""
    for tool in all_tools:
        if getattr(tool, "name", None) == name:
            # beta_async_tool-decorated functions are callable with the right kwargs
            result = await tool.fn(**tool_input)
            return result if isinstance(result, str) else json.dumps(result)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ===========================================================================
# Runtime commands
# ===========================================================================

def _handle_skill_command(
    user_input: str,
    active_skills: list[Skill],
    all_tools: list,
) -> None:
    parts = user_input.split()
    if len(parts) < 2:
        print("\n  Available skills:")
        for name, skill in ALL_SKILLS.items():
            status = "ON" if skill in active_skills else "off"
            print(f"    [{status}] {name} — {skill.description}")
        print("  Usage: /skill enable <name> | /skill disable <name>\n")
        return

    action = parts[1].lower()
    if len(parts) < 3:
        print("  Usage: /skill enable <name> | /skill disable <name>")
        return

    skill_name = parts[2].lower()
    if skill_name not in ALL_SKILLS:
        print(f"  Unknown skill '{skill_name}'. Available: {', '.join(ALL_SKILLS.keys())}")
        return

    skill = ALL_SKILLS[skill_name]
    if action == "enable":
        if skill not in active_skills:
            active_skills.append(skill)
            all_tools.extend(skill.tools)
            print(f"  Enabled skill: {skill_name}")
        else:
            print(f"  Skill '{skill_name}' is already enabled.")
    elif action == "disable":
        if skill in active_skills:
            active_skills.remove(skill)
            for t in skill.tools:
                if t in all_tools:
                    all_tools.remove(t)
            print(f"  Disabled skill: {skill_name}")
        else:
            print(f"  Skill '{skill_name}' is not enabled.")
    else:
        print(f"  Unknown action '{action}'. Use 'enable' or 'disable'.")


def _handle_tools_command(
    all_tools: list,
    active_skills: list[Skill],
    mcp_connections: list,
) -> None:
    print("\n  Available tools:")
    print("  ── Core ──")
    for t in CORE_TOOLS:
        print(f"    • {t.name}")
    for skill in active_skills:
        print(f"  ── Skill: {skill.name} ──")
        for t in skill.tools:
            print(f"    • {t.name}")
    for conn in mcp_connections:
        print(f"  ── MCP: {conn.name} ──")
        for t in conn.tools:
            print(f"    • {t.name}")
    print()


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Stock Quote Analysis Agent")
    parser.add_argument(
        "--skills", type=str, default="",
        help="Comma-separated skills to activate: technical,sector,portfolio",
    )
    parser.add_argument(
        "--mcp-server", nargs=2, action="append", metavar=("NAME", "JSON"),
        help=(
            'Add an MCP server. NAME is a label; JSON is the server config, e.g. '
            '\'{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}\''
        ),
    )
    args = parser.parse_args()

    skill_names = [s.strip() for s in args.skills.split(",") if s.strip()]
    mcp_specs: list[tuple[str, dict]] = []
    for name, spec_json in (args.mcp_server or []):
        try:
            spec = json.loads(spec_json)
            mcp_specs.append((name, spec))
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON for MCP server '{name}': {e}")
            sys.exit(1)

    asyncio.run(run_agent(skill_names, mcp_specs))


if __name__ == "__main__":
    main()
