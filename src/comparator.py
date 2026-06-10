"""
comparator.py — Cross-document comparison.

Sends both extracted data dicts to the LLM and asks it to:
  - Find fields that agree
  - Find fields that conflict (and show both values)
  - Note what is unique to each document
  - Decide whether both documents describe the same product or variants
"""

import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils import parse_llm_json



## system prompt and user prompt template for the comparison node
_SYSTEM = """You are a compliance analyst who verifies consistency across \
manufacturer documents for solar inverter imports.
Always respond with valid JSON only — no prose, no markdown fences."""

_USER_TEMPLATE = """Compare these two sets of data extracted from two manufacturer documents.

Document 1 — {label1}:
{data1}

Document 2 — {label2}:
{data2}

Produce a JSON object with exactly these keys:

  matching_fields    : Object — field_name → shared value (only where both documents agree and neither is "Not found")
  conflicting_fields : List of objects, each with:
                         field_name  (string)
                         doc1_value  (string)
                         doc2_value  (string)
                         note        (brief plain-English explanation of the discrepancy)
  only_in_doc1       : Object — field_name → value (present in Doc1, missing/not found in Doc2)
  only_in_doc2       : Object — field_name → value (present in Doc2, missing/not found in Doc1)
  product_relationship : One of: "same product", "different variants", "different products", "unclear"
  variant_notes      : 2-3 sentence explanation of how the products relate to each other
  summary            : 3-4 sentence plain-English summary of what these documents show and where they agree or disagree

Return ONLY the JSON object."""



## functions to call the LLM for comparison and to be used as a node in the workflow
def compare_extractions(
    llm: ChatGroq,
    pdf1_info: dict,
    pdf2_info: dict,
    pdf1_label: str = "PDF 1",
    pdf2_label: str = "PDF 2",
) -> dict:
    """
    Compare two extraction results and identify matches, conflicts, and gaps.

    Returns a dict with comparison results, or {"error": ...} on failure.
    """
    prompt = _USER_TEMPLATE.format(
        label1=pdf1_label,
        data1=json.dumps(pdf1_info, indent=2),
        label2=pdf2_label,
        data2=json.dumps(pdf2_info, indent=2),
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return parse_llm_json(response.content, fallback_label="document comparison")

    except Exception as exc:
        return {"error": f"LLM comparison call failed: {exc}"}
