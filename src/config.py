"""
config.py — Central configuration and path definitions.
All settings live here so nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

### LLM 
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"   # Fast, large-context Groq model

# ── LangSmith tracing (optional — nice to have for demo) ─────────────────────
os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "AI_Assessment")

### Path definitions
BASE_DIR: Path    = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = BASE_DIR / "dataset"
OUTPUT_DIR: Path  = BASE_DIR / "output"

PDF1_PATH:   Path = DATASET_DIR / "DSS_GZES230100125901_combined-1.pdf"
PDF2_PATH:   Path = DATASET_DIR / "188_1115.pdf"
NEPQA_PATH:  Path = DATASET_DIR / "nepal-photovoltaic-quality-assurance-2025-nepqa-2025.pdf"

### Token-safety limits
# Groq's llama-3.3-70b context window is 128 k tokens.
# We cap raw text going into prompts to leave room for instructions + output.
MAX_PDF_CHARS:   int = 14_000   # ~3 500 tokens — enough for a full datasheet
MAX_NEPQA_CHARS: int = 12_000   # NEPQA guideline excerpt
