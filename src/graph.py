"""
graph.py — LangGraph workflow definition.

Pipeline (sequential nodes):
  load_documents
       ↓
  extract_pdf1  →  extract_pdf2          (one after the other, same LLM client)
       ↓
  compare_docs
       ↓
  map_compliance
       ↓
  generate_report
       ↓
  save_output  →  END

Each node receives the full state dict, does its work, and returns only the keys
it updates.  LangGraph merges those updates into the running state automatically.
"""

import json
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

import src.config as config
from src.pdf_loader import extract_text_from_pdf
from src.extractor import extract_info_from_text
from src.comparator import compare_extractions
from src.compliance_mapper import map_to_nepqa
from src.report_generator import generate_final_report, generate_approach_note


# =============== State schema =======================
# TypedDict keeps the state explicit and readable.
# Every key that a node might read or write must be declared here.

class GraphState(TypedDict):
    # Raw PDF text (populated by load_documents)
    pdf1_raw_text: str
    pdf2_raw_text: str
    nepqa_raw_text: str

    # Structured extractions (populated by extract_pdf1 / extract_pdf2)
    pdf1_info: dict
    pdf2_info: dict

    # Cross-document analysis (populated by compare_docs)
    comparison: dict

    # NEPQA compliance assessment (populated by map_compliance)
    compliance: dict

    # Final written outputs (populated by generate_report)
    final_report: str
    approach_note: str

    # Pipeline error log — nodes append here instead of crashing
    errors: list


# ── LLM client (built once, shared across all nodes) ─────────────────────────

def _build_llm() -> ChatGroq:
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=0,          # 0 = deterministic, best for structured extraction
        groq_api_key=config.GROQ_API_KEY,
    )


# ── Node definitions ──────────────────────────────────────────────────────────

def load_documents(state: GraphState) -> dict:
    """
    Node 1: Load all three PDFs and store their raw text in state.
    No LLM call here — pure file I/O.
    """
    print("\n[1/6] Loading documents...")

    pdf1_text  = extract_text_from_pdf(config.PDF1_PATH,  max_chars=config.MAX_PDF_CHARS)
    pdf2_text  = extract_text_from_pdf(config.PDF2_PATH,  max_chars=config.MAX_PDF_CHARS)
    nepqa_text = extract_text_from_pdf(config.NEPQA_PATH, max_chars=config.MAX_NEPQA_CHARS)

    print(f"     PDF 1 (DSS_GZES...): {len(pdf1_text):,} chars")
    print(f"     PDF 2 (188_1115):    {len(pdf2_text):,} chars")
    print(f"     NEPQA 2025:          {len(nepqa_text):,} chars")

    return {
        "pdf1_raw_text": pdf1_text,
        "pdf2_raw_text": pdf2_text,
        "nepqa_raw_text": nepqa_text,
        "errors": [],
    }


def extract_pdf1(state: GraphState) -> dict:
    """
    Node 2: Use LLM to extract structured technical fields from PDF 1.
    """
    print("\n[2/6] Extracting information from PDF 1...")
    llm  = _build_llm()
    info = extract_info_from_text(llm, state["pdf1_raw_text"], "PDF 1 (DSS_GZES)")

    if "error" in info:
        print(f" {info['error']}")
        return {"pdf1_info": info, "errors": state["errors"] + [info["error"]]}

    print(f" Extracted {len([v for v in info.values() if v != 'Not found in document'])} non-empty fields")
    return {"pdf1_info": info}


def extract_pdf2(state: GraphState) -> dict:
    """
    Node 3: Use LLM to extract structured technical fields from PDF 2.
    """
    print("\n[3/6] Extracting information from PDF 2...")
    llm  = _build_llm()
    info = extract_info_from_text(llm, state["pdf2_raw_text"], "PDF 2 (188_1115)")

    if "error" in info:
        print(f"{info['error']}")
        return {"pdf2_info": info, "errors": state["errors"] + [info["error"]]}

    print(f" Extracted {len([v for v in info.values() if v != 'Not found in document'])} non-empty fields")
    return {"pdf2_info": info}


def compare_docs(state: GraphState) -> dict:
    """
    Node 4: Compare the two extraction results.
    Identifies matching fields, conflicts, and items unique to each document.
    """
    print("\n[4/6] Comparing documents and detecting conflicts...")
    llm = _build_llm()

    comparison = compare_extractions(
        llm,
        state["pdf1_info"],
        state["pdf2_info"],
        pdf1_label="DSS_GZES230100125901",
        pdf2_label="188_1115",
    )

    if "error" in comparison:
        print(f"{comparison['error']}")
        return {"comparison": comparison, "errors": state["errors"] + [comparison["error"]]}

    n_conflicts = len(comparison.get("conflicting_fields", []))
    n_matches   = len(comparison.get("matching_fields", {}))
    print(f"     Matching fields: {n_matches}  |  Conflicts found: {n_conflicts}")
    print(f"     Product relationship: {comparison.get('product_relationship', 'unclear')}")
    return {"comparison": comparison}


def map_compliance(state: GraphState) -> dict:
    """
    Node 5: Map the comparison results against NEPQA 2025 requirements.
    Produces a structured compliance gap analysis.
    """
    print("\n[5/6] Mapping to NEPQA 2025 compliance requirements...")
    llm = _build_llm()

    compliance = map_to_nepqa(llm, state["comparison"], state["nepqa_raw_text"])

    if "error" in compliance:
        print(f"{compliance['error']}")
        return {"compliance": compliance, "errors": state["errors"] + [compliance["error"]]}

    readiness    = compliance.get("overall_readiness", "unknown")
    n_missing    = len(compliance.get("missing_items", []))
    n_priority   = len(compliance.get("priority_gaps", []))
    print(f" Readiness: {readiness}  |  Missing items: {n_missing}  |  Priority gaps: {n_priority}")
    return {"compliance": compliance}


def generate_report(state: GraphState) -> dict:
    """
    Node 6: Generate the final Nepal Import Review Draft document
    and a short approach note.
    """
    print("\n[6/6] Generating final draft document...")
    llm = _build_llm()

    report  = generate_final_report(llm, state["comparison"], state["compliance"])
    note    = generate_approach_note(llm, state["comparison"], state["compliance"])

    print(" Report and approach note generated")
    return {"final_report": report, "approach_note": note}


def save_output(state: GraphState) -> dict:
    """
    Node 7 (final): Write outputs to the /output directory.
      - nepal_import_draft.md  : The document SunBridge shares with their agent
      - analysis_data.json     : All intermediate analysis data (for transparency)
    """
    print("\n[7/7] Saving output files...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Main report ───────────────────────────────────────────────────────────
    report_path = config.OUTPUT_DIR / "nepal_import_draft.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("## Approach Note\n\n")
        f.write(state["approach_note"])
        f.write("\n\n---\n\n")
        f.write(state["final_report"])

    # ── Intermediate analysis data (JSON) ─────────────────────────────────────
    data_path = config.OUTPUT_DIR / "analysis_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf1_extracted_fields": state["pdf1_info"],
                "pdf2_extracted_fields": state["pdf2_info"],
                "cross_document_comparison": state["comparison"],
                "nepqa_compliance_assessment": state["compliance"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"{report_path}")
    print(f"{data_path}")

    if state.get("errors"):
        print(f"\n Pipeline warnings: {state['errors']}")

    return {}


### graph assembly function — defines the nodes and edges of the workflow, and compiles it for execution
def build_graph():
    """
    Assemble and compile the LangGraph StateGraph.

    Returns a compiled graph ready for .invoke().
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("load_documents", load_documents)
    graph.add_node("extract_pdf1",   extract_pdf1)
    graph.add_node("extract_pdf2",   extract_pdf2)
    graph.add_node("compare_docs",   compare_docs)
    graph.add_node("map_compliance", map_compliance)
    graph.add_node("generate_report", generate_report)
    graph.add_node("save_output",    save_output)

    # Sequential edges
    graph.set_entry_point("load_documents")
    graph.add_edge("load_documents",  "extract_pdf1")
    graph.add_edge("extract_pdf1",    "extract_pdf2")
    graph.add_edge("extract_pdf2",    "compare_docs")
    graph.add_edge("compare_docs",    "map_compliance")
    graph.add_edge("map_compliance",  "generate_report")
    graph.add_edge("generate_report", "save_output")
    graph.add_edge("save_output",     END)

    return graph.compile()
