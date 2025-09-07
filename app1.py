import os
import html
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import importlib.util

import streamlit as st
import pandas as pd
import PyPDF2
import pdfplumber

# ------------------------- MUST BE FIRST STREAMLIT CALL -------------------------
st.set_page_config(
    page_title="AI Insights Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- Logging -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
#  Theme bits
# ===========================================================
BELL_BLUE = "#0072CE"
WHITE = "#FFFFFF"

CUSTOM_CSS = f"""
<style>
/* Keep default white background + black text */

/* Hero title bar (kept for light branding) */
.hero-title {{
  background: {BELL_BLUE};
  color: {WHITE};
  padding: 14px 18px;
  border-radius: 10px;
  font-weight: 700;
  font-size: 1.5rem;
  margin-bottom: 8px;
  text-align: center;
}}

/* Download button (blue) */
.stDownloadButton > button {{
  border-radius: 8px !important;
  border: 1px solid {BELL_BLUE} !important;
  background: {BELL_BLUE} !important;
  color: {WHITE} !important;
}}
.stDownloadButton > button:hover {{ filter: brightness(0.95); }}

/* Sidebar spacing */
section[data-testid="stSidebar"] h2 {{ margin-top: 0.5rem; }}

/* Focus list bullets */
ul.focus-list {{
  padding-left: 1.1rem;
  margin-top: 0.25rem;
}}
ul.focus-list li {{ margin: 0.25rem 0; }}

/* Chat bubbles */
.stChatMessage.user {{
  background: rgba(0,0,0,0.03);
  border-radius: 12px;
  padding: 12px 14px;
  border: 1px solid rgba(0,0,0,0.08);
}}
.stChatMessage.assistant {{
  background: rgba(0,0,0,0.02);
  border-radius: 12px;
  padding: 12px 14px;
  border: 1px solid rgba(0,0,0,0.06);
}}

/* Chat input hint */
.chat-subtle {{
  color: #6B7280; /* gray-500 */
  font-size: 0.8rem;
}}

/* Sample questions wrapper (small & subtle) */
.sample-hint-title {{
  color: #6B7280;  /* gray-500 */
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}}

/* Only target sample-question buttons */
.stButton > button[title="sample-question"] {{
  text-align: left;
  width: 100%;
  background: #F8FAFC;
  color: #374151;
  border: 1px solid #E5E7EB;
  border-radius: 10px;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;   /* ← adjust size here */
}}

.stButton > button[title="sample-question"]:hover {{
  background: #EEF2F7;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===========================================================
#  Constants
# ===========================================================
FALLBACK_PROMPT = "Analyze the provided data and provide insights."

# ===========================================================
#  Helpers: load external config & prompt text
# ===========================================================
def load_reports_config() -> Dict[str, Dict[str, Any]]:
    """Load REPORTS_CONFIG from config/reports_config.py."""
    cfg_path = os.path.join("config", "reports_config.py")
    if not os.path.exists(cfg_path):
        logger.error("config/reports_config.py not found.")
        return {}

    try:
        spec = importlib.util.spec_from_file_location("reports_config", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        reports = getattr(mod, "REPORTS_CONFIG", {})
        if not isinstance(reports, dict):
            logger.error("REPORTS_CONFIG missing or invalid in config/reports_config.py")
            return {}
        return reports
    except Exception as e:
        logger.error(f"Failed to load reports_config.py: {e}")
        return {}

def read_prompt_text(prompt_path: Optional[str]) -> str:
    """Read prompt text from file; fallback when missing/empty."""
    try:
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                return text if text else FALLBACK_PROMPT
    except Exception as e:
        logger.warning(f"Read prompt failed {prompt_path}: {e}")
    return FALLBACK_PROMPT

# ===========================================================
#  AI Client (Gemini via Vertex)
# ===========================================================
class AIClient:
    """Initialize the connection to Gemini. App still works if not configured."""
    def __init__(self):
        self.available = False
        try:
            from google import genai
            self.client = genai.Client(
                vertexai=True,
                project='prj-exp-sbx-apr-xqs7fz4g',
                location='us-central1'
            )
            self.available = True
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            st.warning("AI client not configured")

    def generate_content(self, prompt: str) -> str:
        """Call the model to generate content."""
        if not self.available:
            return "AI client unavailable."
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt],
                config={"temperature": 0.0}
            )
            return response.text
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return f"Unable to generate AI insights: {e}"

# ===========================================================
#  Analyzer
# ===========================================================
class ReportAnalyzer:
    """Handles data loading/extraction and AI generation."""

    def __init__(self, reports_config: Dict[str, Dict[str, Any]]):
        self.ai_client = AIClient()
        self.reports_config = reports_config

    def extract_pdf_text(self, file_path: str) -> str:
        """Extract PDF text (no truncation)."""
        try:
            text_content = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
            if not text_content.strip():
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += (page.extract_text() or "") + "\n\n"
            return text_content.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return f"Error extracting PDF content: {e}"

    @st.cache_data
    def load_report_data(_self, file_path: str, file_type: str) -> Optional[pd.DataFrame]:
        """Load data for the selected report (not previewed; only for AI context)."""
        try:
            if not os.path.exists(file_path):
                st.error(f"File {file_path} not found. Please add the file to continue.")
                return None

            if file_type == 'pdf':
                pdf_text = _self.extract_pdf_text(file_path)
                return pd.DataFrame({"Content": [pdf_text]})

            if file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path, header=0)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            st.error(f"Error loading file: {e}")
            return None

    def create_file_download(self, file_path: str, file_type: str) -> Tuple[bytes, str]:
        """Provide raw file bytes for download, with MIME."""
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
            mime = "application/pdf" if file_type == 'pdf' else \
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return raw, mime
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            st.error(f"Error reading file for download: {e}")
            return b"", "application/octet-stream"

    def generate_ai_content(self, unified_prompt: str, data_context: str = "", question_context: str = "") -> str:
        """Generate AI output (no sentence limit)."""
        parts = [unified_prompt]
        if question_context:
            parts.append(f"\nSpecific Context: {question_context}")
        parts.append(f"\nData:\n{data_context}")
        return self.ai_client.generate_content("".join(parts))

# ===========================================================
#  UI
# ===========================================================
def main():
    reports_config = load_reports_config()
    if not reports_config:
        st.error("No reports found in config/reports_config.py. Please check REPORTS_CONFIG.")
        return

    analyzer = ReportAnalyzer(reports_config)

    # Sidebar
    with st.sidebar:
        st.header("📋 Report")
        selected_report = st.selectbox(
            "Choose a report:",
            list(analyzer.reports_config.keys()),
            help="Each report uses a unified external prompt"
        )
        report_cfg = analyzer.reports_config[selected_report]

        st.subheader("Report Info")
        st.write(report_cfg.get("description", ""))

        st.subheader("Focus Areas")
        focus = report_cfg.get("focus_areas", [])
        if focus:
            st.markdown(
                "<ul class='focus-list'>" +
                "".join([f"<li>{html.escape(x)}</li>" for x in focus]) +
                "</ul>",
                unsafe_allow_html=True
            )

    # Main area
    st.markdown("<div class='hero-title'>🤖 Welcome to Customer Ops AI Analyst – Your Data, Your Dialogue</div>", unsafe_allow_html=True)

    # Download
    file_path = report_cfg.get("file", "")
    file_type = report_cfg.get("file_type", "excel")
    file_bytes, mime = analyzer.create_file_download(file_path, file_type)
    ext = "pdf" if file_type == "pdf" else "xlsx"
    st.download_button(
        label=f"📥 Download {selected_report} Report",
        data=file_bytes,
        file_name=f"{selected_report.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}",
        mime=mime,
        help="Download the original report file"
    )

    st.divider()

    # Load data (for AI context only)
    df = analyzer.load_report_data(file_path, file_type)

    # ---------------- Chat FIRST (highlight the core) ----------------
    st.subheader("Chat with the report")
    st.caption('<span class="chat-subtle">Ask direct questions. The assistant will cite the report content implicitly.</span>', unsafe_allow_html=True)
    chat_input = st.chat_input(f"Ask about {selected_report}...")

    # ---------------- Sample Questions BELOW chat ----------------
    sample_questions = report_cfg.get("sample_questions", [])
    if isinstance(sample_questions, dict):
        iter_items = list(sample_questions.items())
    else:
        iter_items = [(item.get("q", ""), {"context": item.get("context", "")}) for item in sample_questions]

    if iter_items:
        st.markdown('<div class="sample-hint-title">Quick suggestions (optional):</div>', unsafe_allow_html=True)
        with st.container():
            cols = st.columns(2)
            for i, (q, cfg) in enumerate(iter_items):
                if not q:
                    continue
                with cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", help="sample-question"):
                        st.session_state.user_question = q
                        st.session_state.question_config = cfg or {}

    # Decide the effective user input after possible sample click
    if hasattr(st.session_state, "user_question"):
        user_input = st.session_state.user_question
        question_config = getattr(st.session_state, "question_config", {})
        delattr(st.session_state, "user_question")
        if hasattr(st.session_state, "question_config"):
            delattr(st.session_state, "question_config")
    else:
        user_input = chat_input
        question_config = {}

    # Respond with AI
    if user_input and df is not None:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                if file_type == "pdf":
                    data_context = df["Content"].iloc[0] if "Content" in df.columns else "No content available"
                else:
                    data_context = df.to_string(index=False)

                unified_prompt = read_prompt_text(report_cfg.get("prompt_file"))
                q_ctx = question_config.get("context", "") if isinstance(question_config, dict) else ""

                response = analyzer.generate_ai_content(
                    unified_prompt=unified_prompt,
                    data_context=data_context,
                    question_context=q_ctx
                )
                st.write(response)

    elif user_input and df is None:
        with st.chat_message("assistant"):
            st.write("Please ensure the report file is available before asking questions.")

if __name__ == "__main__":
    main()


