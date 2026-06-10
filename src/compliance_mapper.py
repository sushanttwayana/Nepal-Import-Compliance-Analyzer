"""
compliance_mapper.py — Map comparison results against NEPQA 2025 requirements.

Uses the NEPQA 2025 guideline text as context to check:
  - What Nepal's import review typically asks for
  - What is already available from the manufacturer documents
  - What is missing or conflicting
  - Overall readiness level
"""

import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils import parse_llm_json


## System prompt and user prompt template for the compliance mapping node

_SYSTEM = """You are a Nepal import compliance specialist with knowledge of the \
NEPQA 2025 quality assurance framework for photovoltaic equipment.
Always respond with valid JSON only — no prose, no markdown fences."""

_USER_TEMPLATE = """Using the NEPQA 2025 guideline excerpt below, assess the Nepal \
import compliance status for a solar inverter shipment from China.

--- NEPQA 2025 GUIDELINE EXCERPT ---
{nepqa_text}
--- END GUIDELINE ---

--- PRODUCT INFORMATION (from cross-document analysis) ---
{comparison_summary}
--- END PRODUCT INFORMATION ---

Produce a JSON object with exactly these keys:

  required_items      : List of strings — items Nepal import review typically requires
                        for grid-tied inverters, based on the NEPQA 2025 guideline
  available_items     : List of objects, each with:
                          item        (what is required)
                          status      ("available" | "partial" | "conflicting")
                          source      (which document / where it was found)
                          value       (the actual value or note)
  missing_items       : List of strings — required items with no available data
  conflicting_items   : List of objects, each with:
                          item        (field or document type)
                          doc1_value  (value from PDF 1)
                          doc2_value  (value from PDF 2)
                          impact      (why this matters for Nepal import)
  labeling_assessment : String — assessment of labeling and marking requirements
                        vs. what is available
  certification_assessment : String — assessment of certifications available
                             vs. what Nepal would expect
  overall_readiness   : One of: "ready", "partial", "not ready"
  readiness_reason    : 2-3 sentence explanation of the readiness verdict
  priority_gaps       : List of the top 3-5 most critical missing items
                        SunBridge must resolve before final filing
  notes               : Any other observations relevant to Nepal import compliance

Return ONLY the JSON object."""


### Function to call the LLM for compliance mapping, to be used as a node in the workflow
def map_to_nepqa(llm: ChatGroq, comparison: dict, nepqa_text: str) -> dict:
    """
    Assess Nepal import compliance using NEPQA 2025 guideline as reference.

    Args:
        llm:         Initialised ChatGroq client.
        comparison:  Output dict from comparator.compare_extractions().
        nepqa_text:  Raw text extracted from the NEPQA 2025 PDF.

    Returns:
        dict with compliance assessment, or {"error": ...} on failure.
    """
    # Keep prompts within token limits
    nepqa_excerpt     = nepqa_text[:12_000]
    comparison_excerpt = json.dumps(comparison, indent=2)[:6_000]

    prompt = _USER_TEMPLATE.format(
        nepqa_text=nepqa_excerpt,
        comparison_summary=comparison_excerpt,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return parse_llm_json(response.content, fallback_label="NEPQA compliance mapping")

    except Exception as exc:
        return {"error": f"LLM compliance mapping call failed: {exc}"}
