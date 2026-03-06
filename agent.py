"""
Simple agent using the Claude API with tool use.

The agent can answer questions using two tools:
  - get_weather: returns fake weather data for a city
  - calculator: evaluates basic arithmetic expressions
"""

import anthropic
import ast
import operator

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. Paris",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression (supports +, -, *, /).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate, e.g. '3 * (4 + 2)'",
                }
            },
            "required": ["expression"],
        },
    },
]

# Fake weather data
WEATHER_DATA = {
    "paris": "Sunny, 22°C",
    "london": "Cloudy, 15°C",
    "new york": "Partly cloudy, 18°C",
    "tokyo": "Clear, 25°C",
}

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression."""
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {node.op}")
            return op(_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {node}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _eval(tree.body)


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "get_weather":
        city = tool_input["city"].lower()
        return WEATHER_DATA.get(city, f"No weather data available for '{tool_input['city']}'.")
    if name == "calculator":
        try:
            result = _safe_eval(tool_input["expression"])
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"
    return f"Unknown tool: {name}"


def run_agent(user_message: str) -> str:
    """Run the agentic loop until Claude produces a final answer."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Return the last text block
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Claude wants to call tools
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # Append assistant turn (including tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for tool in tool_use_blocks:
            result = execute_tool(tool.name, tool.input)
            print(f"  [tool] {tool.name}({tool.input}) → {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool.id,
                    "content": result,
                }
            )

        # Feed results back to Claude
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    questions = [
        "What's the weather like in Paris and Tokyo?",
        "What is (123 + 456) * 2?",
        "If it's 22°C in Paris, convert that to Fahrenheit using the formula F = C * 9/5 + 32.",
    ]

    for question in questions:
        print(f"\nUser: {question}")
        answer = run_agent(question)
        print(f"Agent: {answer}")
