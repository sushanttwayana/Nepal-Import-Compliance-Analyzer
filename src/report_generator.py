"""
report_generator.py — Generates two outputs:
  1. The main Nepal Import Review Draft (Markdown document)
  2. A short approach note explaining how the analysis was done

The draft is what SunBridge Trading hands to their import agent.
It should be honest, plain, and clearly mark anything that conflicts or is missing.
"""

import json
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

### Prompots

_REPORT_SYSTEM = """You are preparing an import review draft document for SunBridge Trading \
to share with their Nepal import agent.
Write in clear, professional English.
Be factual and honest — this is a working draft, not a final certificate.
Use Markdown formatting."""

_REPORT_USER_TEMPLATE = """Generate a Nepal Import Review Draft Document for SunBridge Trading
based on the analysis below.

PRODUCT COMPARISON ANALYSIS:
{comparison}

NEPQA 2025 COMPLIANCE ASSESSMENT:
{compliance}

---
The document MUST include these sections in order:

1. DOCUMENT HEADER
   - Reference: SUNBRIDGE-NEPAL-{ref_date}
   - Date: {today}
   - Prepared for: SunBridge Trading, Kathmandu
   - Subject: Grid-Tied Solar Inverter — Nepal Import Review Draft
   - Status: WORKING DRAFT — not for final filing

2. PRODUCT IDENTIFICATION
   - What product(s) the documents appear to describe
   - Model number(s) and any variant uncertainty
   - Whether the two source documents appear to cover the same product

3. MANUFACTURER INFORMATION
   - Company name, address
   - Note any conflicts between the two documents with [CONFLICT] tag

4. TECHNICAL SPECIFICATIONS TABLE
   - Markdown table with columns: Parameter | PDF 1 Value | PDF 2 Value | Status
   - Status column: "Match" | "[CONFLICT]" | "[MISSING]"
   - Include: rated power, voltage ranges, frequency, efficiency, IP class,
     temperature range, dimensions, weight, MPPT info

5. CERTIFICATIONS & TEST STANDARDS
   - List what is documented
   - Mark unverified verbal claims with [UNVERIFIED]
   - Mark missing certificates with [MISSING — required for Nepal import]

6. LABELING & MARKING
   - What labeling info is available
   - Mark gaps with [MISSING]

7. NEPAL IMPORT COMPLIANCE NOTES (NEPQA 2025 Reference)
   - What aligns with Nepal import expectations
   - What is flagged as a gap or concern
   - Overall readiness verdict

8. OUTSTANDING ITEMS & NEXT STEPS
   - Numbered list of what SunBridge must resolve before final filing
   - Keep this practical and actionable

---
Formatting rules:
  - Use [CONFLICT: Doc1 says X / Doc2 says Y] for mismatches
  - Use [MISSING — required for Nepal import] for required gaps
  - Use [UNVERIFIED — stated verbally only] for unconfirmed claims
  - Do NOT invent values. If something is not in the data, say so.

Generate the complete document now."""


_APPROACH_SYSTEM = "You write brief, plain-English operational notes for non-technical readers."

_APPROACH_USER_TEMPLATE = """Write a brief approach note (150-200 words) for SunBridge Trading.

Explain:
1. What documents were reviewed and how (mention two manufacturer PDFs + NEPQA 2025 guideline)
2. Key finding about whether the documents describe the same product or variants
3. Top 2-3 issues found
4. What they should do next

Key facts to include:
  Product relationship  : {product_relationship}
  Variant notes         : {variant_notes}
  Overall readiness     : {overall_readiness}
  Priority gaps         : {priority_gaps}

Keep it conversational and practical. No technical jargon. No bullet points — write in short paragraphs."""


def generate_final_report(llm: ChatGroq, comparison: dict, compliance: dict) -> str:
    """Generate the full Nepal Import Review Draft as a Markdown string."""
    today = datetime.now().strftime("%Y-%m-%d")
    ref_date = datetime.now().strftime("%Y%m%d")

    prompt = _REPORT_USER_TEMPLATE.format(
        comparison=json.dumps(comparison, indent=2)[:6_000],
        compliance=json.dumps(compliance, indent=2)[:4_000],
        today=today,
        ref_date=ref_date,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_REPORT_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as exc:
        return f"[ERROR] Report generation failed: {exc}"


def generate_approach_note(llm: ChatGroq, comparison: dict, compliance: dict) -> str:
    """Generate a short plain-English approach note for Ramesh / SunBridge."""
    prompt = _APPROACH_USER_TEMPLATE.format(
        product_relationship=comparison.get("product_relationship", "unclear"),
        variant_notes=comparison.get("variant_notes", "Not determined"),
        overall_readiness=compliance.get("overall_readiness", "Not assessed"),
        priority_gaps=compliance.get("priority_gaps", []),
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_APPROACH_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as exc:
        return f"[ERROR] Approach note generation failed: {exc}"
