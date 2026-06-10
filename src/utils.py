"""
utils.py — Small shared helpers used across modules.
"""

import json
import re


def parse_llm_json(raw: str, fallback_label: str = "LLM response") -> dict:
    """
    Try to parse a JSON object from raw LLM output.

    LLMs sometimes wrap JSON in markdown fences (```json ... ```) or add
    a short sentence before/after. This function handles those cases.

    Returns a dict on success, or {"error": ..., "raw_response": ...} on failure.
    """
    # 1. Direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences then try again
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find the first {...} block (greedy, handles nested braces)
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "error": f"Could not parse JSON from {fallback_label}",
        "raw_response": raw[:500],  # Store first 500 chars for debugging
    }
