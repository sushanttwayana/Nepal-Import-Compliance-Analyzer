"""
main.py — Entry point.

Run:
    python main.py

What happens:
    1. All three PDFs in /dataset are loaded and text is extracted.
    2. Both manufacturer PDFs are analysed by the LLM to extract key fields.
    3. The two extractions are compared — matches and conflicts are identified.
    4. Results are mapped against NEPQA 2025 import requirements.
    5. A Nepal Import Review Draft document is written to /output.
"""

import sys
from dotenv import load_dotenv

load_dotenv()

import src.config as config
from src.graph import build_graph, GraphState


def _check_env() -> bool:
    """Quick sanity check before spending API credits."""
    if not config.GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY is not set in your .env file.")
        print("        Add: GROQ_API_KEY=gsk_...")
        return False

    missing_pdfs = [
        p for p in [config.PDF1_PATH, config.PDF2_PATH, config.NEPQA_PATH]
        if not p.exists()
    ]
    if missing_pdfs:
        print("[ERROR] The following required files are missing from /dataset:")
        for p in missing_pdfs:
            print(f"        - {p.name}")
        return False

    return True


def main() -> None:
    print("=" * 62)
    print("  SunBridge Trading — Nepal Import Compliance Analyzer")
    print("  Powered by LangGraph + Groq (llama-3.3-70b-versatile)")
    print("=" * 62)

    if not _check_env():
        sys.exit(1)

    # Empty initial state — nodes will fill it in as the graph executes
    initial_state: GraphState = {
        "pdf1_raw_text":  "",
        "pdf2_raw_text":  "",
        "nepqa_raw_text": "",
        "pdf1_info":      {},
        "pdf2_info":      {},
        "comparison":     {},
        "compliance":     {},
        "final_report":   "",
        "approach_note":  "",
        "errors":         [],
    }

    graph = build_graph()

    print("\nStarting analysis pipeline...\n")
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 62)
    print("Analysis complete!")
    print("=" * 62)
    print("\nOutput files written to /output:")
    print("  → nepal_import_draft.md    (share this with the Nepal import agent)")
    print("  → analysis_data.json       (full intermediate analysis — for reference)")

    if final_state.get("errors"):
        print(f"\nWarnings during run: {final_state['errors']}")


if __name__ == "__main__":
    main()
