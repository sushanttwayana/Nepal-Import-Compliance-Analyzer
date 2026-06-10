"""
extractor.py — Uses Groq LLM to pull structured technical fields
               from raw manufacturer PDF text.

What it does:
  Sends the raw PDF text to the LLM with a precise extraction prompt.
  The LLM returns a JSON object with all key fields needed for a Nepal
  import review (specs, certifications, labeling, manufacturer info, etc.)
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils import parse_llm_json


############  Prompts 

_SYSTEM = """You are a technical document analyst specialising in solar inverter \
manufacturer documentation.
Your job is to extract structured information from raw PDF text.
Always respond with valid JSON only — no prose, no markdown fences, no extra text."""

_USER_TEMPLATE = """Extract the following fields from this solar inverter manufacturer document.
If a field cannot be found in the text, set its value to "Not found in document".

Fields to extract:
  product_name          : Full product name or series name
  model_number          : Model number(s) — list all if multiple
  manufacturer_name     : Company / manufacturer name
  manufacturer_address  : Full address of manufacturer
  rated_power_ac        : AC output power (W or kW)
  rated_power_dc        : DC input power if stated
  mppt_voltage_range    : MPPT voltage operating range (V)
  max_dc_input_voltage  : Maximum DC input voltage (V)
  number_of_mppt        : Number of MPPT trackers
  ac_output_voltage     : AC output voltage (V)
  ac_frequency          : AC output frequency (Hz)
  max_ac_output_current : Maximum AC output current (A)
  peak_efficiency       : Peak / maximum efficiency (%)
  european_efficiency   : European / weighted efficiency if stated (%)
  thd                   : Total Harmonic Distortion if stated
  ip_protection_class   : IP rating (e.g. IP65)
  operating_temperature : Operating temperature range (°C)
  storage_temperature   : Storage temperature range if stated (°C)
  dimensions            : Physical dimensions (mm)
  weight                : Weight (kg)
  certifications        : All certifications and marks mentioned — return as a list
  test_standards        : All testing standards referenced — return as a list
  grid_standards        : Grid connection standards (e.g. IEEE, IEC, VDE) — list
  protection_features   : Protection features listed (e.g. OVP, OCP, anti-islanding)
  display_interface     : Display or communication interface (LCD, RS485, WiFi, etc.)
  labeling_info         : Any labeling, nameplate, or marking information
  warranty              : Warranty details
  country_of_origin     : Country of manufacture
  document_type         : What kind of document this appears to be (datasheet, test report, certificate, etc.)

---
DOCUMENT TEXT:
{text}
---

Respond with ONLY a JSON object using the field names above as keys."""

# ================================================================================
def extract_info_from_text(llm: ChatGroq, raw_text: str, doc_label: str) -> dict:
    """
    Ask the LLM to extract structured fields from a manufacturer document.

    Args:
        llm:       Initialised ChatGroq client.
        raw_text:  Raw text extracted from the PDF.
        doc_label: Human-readable label used in error messages (e.g. "PDF 1").

    Returns:
        dict with extracted fields, or dict with "error" key if parsing failed.
    """
    prompt = _USER_TEMPLATE.format(text=raw_text)

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return parse_llm_json(response.content, fallback_label=f"{doc_label} extraction")

    except Exception as exc:
        return {"error": f"LLM call failed for {doc_label}: {exc}"}
