# SunBridge Trading — Nepal Import Compliance Analyzer

> **Task 1 | Cantordust AI Engineer Assessment**
> China → Nepal | Grid-Tied Solar Inverter Import Review

---

## What This Project Does

SunBridge Trading received **two manufacturer PDFs** from a Chinese supplier for grid-tied solar inverters. They need a compliance draft to share with their Nepal import agent before final filing.

This tool automates that process:

1. Extracts technical information from both manufacturer PDFs using an LLM
2. Cross-references the two documents and flags every conflict or mismatch
3. Maps the findings against the **NEPQA 2025** (Nepal Photovoltaic Quality Assurance) guideline
4. Generates a clean **Nepal Import Review Draft** document — ready to hand to the import agent

---

## System Architecture

```
main.py
   │
   └─▶  LangGraph Pipeline (src/graph.py)
           │
           ├─ [Node 1] load_documents      ── pdfplumber extracts raw text from all 3 PDFs
           │
           ├─ [Node 2] extract_pdf1        ── Groq LLM extracts 25 structured fields from PDF 1
           │
           ├─ [Node 3] extract_pdf2        ── Groq LLM extracts 25 structured fields from PDF 2
           │
           ├─ [Node 4] compare_docs        ── LLM compares both extractions, finds conflicts
           │
           ├─ [Node 5] map_compliance      ── LLM maps results against NEPQA 2025 requirements
           │
           ├─ [Node 6] generate_report     ── LLM writes the final Import Review Draft
           │
           └─ [Node 7] save_output         ── Saves .md report + .json analysis to /output
```

### Why LangGraph?

LangGraph gives a clean, inspectable state machine. Each node receives the full pipeline state and returns only what it changed — making debugging straightforward and the flow easy to follow or extend.

---

## Project Structure

```
AI_Assessment/
├── dataset/                          ← Input documents (PDFs)
│   ├── DSS_GZES230100125901_combined-1.pdf   (Manufacturer PDF 1)
│   ├── 188_1115.pdf                          (Manufacturer PDF 2)
│   └── nepal-photovoltaic-quality-assurance-2025-nepqa-2025.pdf  (NEPQA 2025)
│
├── src/
│   ├── config.py             ← All paths, model name, token limits
│   ├── pdf_loader.py         ← PDF text extraction (pdfplumber)
│   ├── extractor.py          ← LLM prompt: extract fields from one PDF
│   ├── comparator.py         ← LLM prompt: compare two extractions
│   ├── compliance_mapper.py  ← LLM prompt: map results to NEPQA 2025
│   ├── report_generator.py   ← LLM prompt: write final draft + approach note
│   ├── graph.py              ← LangGraph StateGraph (all nodes + edges)
│   └── utils.py              ← JSON parsing helper
│
├── output/                   ← Generated files (created on first run)
│   ├── nepal_import_draft.md      ← Main output — share this with the import agent
│   └── analysis_data.json         ← Full intermediate data for reference
│
├── main.py                   ← Entry point: run this
├── requirements.txt
├── .env                      ← Your API keys (not committed)
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AI_Assessment
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your `.env` file

Create a `.env` file in the project root (or edit the existing one):

```env
GROQ_API_KEY=gsk_your_key_here

# Optional — enables LangSmith tracing to inspect the pipeline run
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key_here
LANGCHAIN_PROJECT=AI_Assessment
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Place the dataset files

Make sure these three files are inside the `/dataset` folder:

| File | Description |
|---|---|
| `DSS_GZES230100125901_combined-1.pdf` | Manufacturer PDF 1 |
| `188_1115.pdf` | Manufacturer PDF 2 |
| `nepal-photovoltaic-quality-assurance-2025-nepqa-2025.pdf` | NEPQA 2025 guideline |

---

## Run

```bash
python main.py
```

## Output Files

| File | Purpose |
|---|---|
| `output/nepal_import_draft.md` | The main deliverable — share this with the Nepal import agent. Contains product info, specs table, certifications, labeling notes, NEPQA compliance summary, and outstanding items list. |
| `output/analysis_data.json` | Full intermediate JSON — extracted fields from each PDF, comparison results, compliance assessment. Useful for debugging or further processing. |

### What the draft document looks like

The draft follows this structure:

```
Approach Note          ← plain-English summary of how this was prepared
─────────────────────────────────────────────────────────────
1. Document Header     ← reference number, date, status: WORKING DRAFT
2. Product ID          ← what product(s) this appears to be; variant uncertainty
3. Manufacturer Info   ← company, address — conflicts marked [CONFLICT]
4. Technical Specs     ← table: Parameter | PDF 1 | PDF 2 | Status
5. Certifications      ← what exists, what's missing [MISSING]
6. Labeling            ← what's known, what needs photos [MISSING]
7. NEPQA Compliance    ← per NEPQA 2025 reference — what aligns, what's flagged
8. Outstanding Items   ← numbered action list for SunBridge
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| **Groq + llama-3.3-70b-versatile** | Fast inference, large (128k) context window, free-tier available |
| **pdfplumber for PDF parsing** | Better at multi-column layouts than PyPDF2; handles page structure cleanly |
| **Sequential LangGraph nodes** | Easier to read, debug, and demo vs. parallel execution for this pipeline size |
| **JSON-only LLM responses** | Structured output makes downstream processing reliable; `utils.parse_llm_json` handles edge cases |
| **Token limits in config.py** | Prevents hitting Groq rate limits; `MAX_PDF_CHARS` and `MAX_NEPQA_CHARS` are easily tunable |
| **Errors list in state** | Pipeline continues even if one node partially fails — partial output is better than no output |

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | Groq API (`langchain-groq`) |
| Agent workflow | LangGraph (`langgraph`) |
| PDF extraction | pdfplumber |
| Environment | python-dotenv |

---