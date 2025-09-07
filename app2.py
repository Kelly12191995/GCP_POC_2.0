import os
import html
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import importlib.util

import streamlit as st
import pandas as pd
import PyPDF2
import pdfplumber
import tempfile

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None

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

/* Only target sample-question buttons (using help -> title attribute) */
.stButton > button[title="sample-question"] {{
  text-align: left;
  width: 100%;
  background: #F8FAFC;
  color: #374151;
  border: 1px solid #E5E7EB;
  border-radius: 10px;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;   /* Adjust size here */
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
#  Topic routing helpers (Plan C: dual-source evidence with tags)
# ===========================================================
def topic_match(user_text: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in user_text (case-insensitive, naive contains)."""
    if not user_text or not keywords:
        return False
    t = user_text.lower()
    for kw in keywords:
        kw = (kw or "").lower().strip()
        if not kw:
            continue
        if kw in t:
            return True
    return False

@st.cache_data
def load_supplement_excel(path: str, sheet: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load the supplement Excel (cached)."""
    try:
        if not path or not os.path.exists(path):
            return None
        if sheet and sheet != "WE":
            df = pd.read_excel(path, sheet_name=sheet)
        else:
            # 'latest': default to the first sheet
            df = pd.read_excel(path)
        return df
    except Exception as e:
        logger.exception(f"Failed to load supplement Excel: {e}")
        return None

def format_sources(confluence_path: Optional[str], report_file: Optional[str]) -> str:
    """Build a Sources line with last-updated info from file mtimes."""
    parts = []
    try:
        if confluence_path and os.path.exists(confluence_path):
            ts = datetime.fromtimestamp(os.path.getmtime(confluence_path)).strftime("%Y-%m-%d")
            parts.append(f"Confluence CCTS (internal Excel; last updated: {ts})")
    except Exception:
        pass
    try:
        if report_file and os.path.exists(report_file):
            ts = datetime.fromtimestamp(os.path.getmtime(report_file)).strftime("%Y-%m-%d")
            parts.append(f"Weekly Merged Report (file date: {ts})")
    except Exception:
        pass
    return ("Sources: " + "; ".join(parts)) if parts else ""

def build_dual_source_pack(
    ccts_df: pd.DataFrame,
    weekly_blob: str,
    max_rows: int = 100
) -> str:
    """
    Build a dual-source evidence pack with explicit tags.
    - [CCTS] section: top rows of the CCTS Excel encoded as CSV.
    - [Weekly] section: raw weekly report string (pdf text / table string).
    """
    trimmed = ccts_df.head(max_rows)
    ccts_csv = trimmed.to_csv(index=False)
    pack = (
        "CCTS coverage in the weekly report is limited. "
        "Use the Confluence-based CCTS dataset as the primary source for CCTS-related facts.\n\n"
        "[CCTS] Evidence (CSV):\n"
        f"{ccts_csv}\n\n"
        "[Weekly] Evidence (raw):\n"
        f"{weekly_blob}"
    )
    return pack

def ccts_answer_style_instruction() -> str:
    """
    Guidance injected when CCTS topic is detected (Plan C).
    Forces inline evidence tagging and executive-friendly clarity.
    """
    return (
        "When the topic concerns CCTS:\n"
        "- Lead with the main finding in one concise sentence.\n"
        "- For each key fact, append an inline evidence tag: [CCTS] or [Weekly].\n"
        "- Prefer [CCTS] when sources diverge; explicitly call out discrepancies.\n"
        "- Be specific with dates/ranges and counts; avoid vague wording.\n"
        "- End your answer with a compact 'Sources' line if provided in the data."
    )

# ===========================================================
#  Conversation helpers
# ===========================================================
def conversation_to_text(conversation: List[Dict[str, str]]) -> str:
    """Serialize conversation messages into plain text."""
    lines = []
    for msg in conversation:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n\n".join(lines)

def conversation_to_pdf(conversation: List[Dict[str, str]]) -> Optional[bytes]:
    """Return PDF bytes for the conversation; requires fpdf."""
    if FPDF is None:
        return None
    text = conversation_to_text(conversation)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest="S").encode("latin-1")

def send_conversation_email(
    to_email: str,
    conversation: List[Dict[str, str]],
    pdf_bytes: Optional[bytes],
    report_path: str,
    subject: str = "Conversation History",
    body: str = "Please see attached conversation and report.",
) -> str:
    """Email conversation and selected report using local Outlook."""
    try:
        import win32com.client  # type: ignore
    except Exception:
        return "Outlook is not available on this system."

    text = conversation_to_text(conversation)

    try:
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.To = to_email
        mail.Subject = subject
        mail.Body = body
        mail.SentOnBehalfOfName = "aidigitaladvocacyplatform@bell.ca"

        # Attach conversation TXT
        txt_tmp = None
        pdf_tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tf:
                tf.write(text.encode("utf-8"))
                txt_tmp = tf.name
            mail.Attachments.Add(txt_tmp)

            if pdf_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pf:
                    pf.write(pdf_bytes)
                    pdf_tmp = pf.name
                mail.Attachments.Add(pdf_tmp)

            if report_path and os.path.exists(report_path):
                mail.Attachments.Add(os.path.abspath(report_path))

            mail.Send()
            return "Email sent."
        finally:
            if txt_tmp and os.path.exists(txt_tmp):
                os.unlink(txt_tmp)
            if pdf_tmp and os.path.exists(pdf_tmp):
                os.unlink(pdf_tmp)
    except Exception as e:
        return f"Failed to send email: {e}"

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
                project='prj-exp-sbx-apr-x564ukcc',
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
                model='gemini-2.0-flash-001',
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

    def generate_ai_content(
        self,
        unified_prompt: str,
        question: str,
        data_context: str = "",
        question_context: str = "",
    ) -> str:
        """Generate AI output (no sentence limit)."""
        parts = [unified_prompt]
        parts.append(f"\nQuestion: {question}")
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

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

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
                "<ul class='focus-list'>"
                + "".join([f"<li>{html.escape(x)}</li>" for x in focus])
                + "</ul>",
                unsafe_allow_html=True
            )

    # Main area
    st.markdown("<div class='hero-title'>🤖 Welcome to Customer Ops AI Analyst – Your Data, Your Dialogue</div>", unsafe_allow_html=True)

    # Download (original report file)
    file_path = report_cfg.get("file", "")
    file_type = report_cfg.get("file_type", "excel")
    file_bytes, mime = analyzer.create_file_download(file_path, file_type)
    ext = "pdf" if file_type == "pdf" else "xlsx"
    st.download_button(
        label=f"📥 Download {selected_report} Report",
        data=file_bytes,
        file_name=f"{selected_report.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}",
        mime=mime,
        help="Download the original report file",
    )

    st.divider()

    # Load data (for AI context only)
    df = analyzer.load_report_data(file_path, file_type)

    # ---------------- Chat FIRST (highlight the core) ----------------
    st.subheader("Chat with the report")
    st.caption('<span class="chat-subtle">Ask direct questions. The assistant will cite the report content implicitly.</span>',
               unsafe_allow_html=True)

    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    chat_input = st.chat_input(f"Ask about {selected_report}...")

    # ---------------- Sample Questions BELOW chat ----------------
    sample_questions = report_cfg.get("sample_questions", [])
    if isinstance(sample_questions, dict):
        # Backward-compatible (old dict format)
        iter_items = list(sample_questions.items())
    else:
        # New list-of-dicts format
        iter_items = [(item.get("q", ""), {"context": item.get("context", "")}) for item in sample_questions]

    if iter_items:
        st.markdown('<div class="sample-hint-title">Quick suggestions (optional):</div>', unsafe_allow_html=True)
        with st.container():
            cols = st.columns(2)
            for i, (q, cfg) in enumerate(iter_items):
                if not q:
                    continue
                with cols[i % 2]:
                    # IMPORTANT: add help="sample-question" for CSS targeting
                    if st.button(q, key=f"sq_{i}", help="sample-question"):
                        st.session_state.user_question = q
                        st.session_state.question_config = cfg or {}

    # Decide effective user input after sample click
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
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                # Build base data_context from the selected report
                if file_type == "pdf":
                    data_context = df["Content"].iloc[0] if "Content" in df.columns else "No content available"
                else:
                    data_context = df.to_string(index=False)

                unified_prompt = read_prompt_text(report_cfg.get("prompt_file"))
                q_ctx = question_config.get("context", "") if isinstance(question_config, dict) else ""

                # --------- Plan C: dual-source evidence for CCTS ----------
                final_context = data_context  # fallback: just the weekly report
                supplements = report_cfg.get("supplements", [])
                used_supplement = None
                supplement_df = None
                extra_instruction = ""

                if supplements and user_input:
                    for sup in supplements:
                        if (sup.get("topic", "").lower() == "ccts" and
                            topic_match(user_input, sup.get("keywords", []))):
                            used_supplement = sup
                            supplement_df = load_supplement_excel(
                                sup.get("file"),
                                sup.get("sheet")
                            )
                            break

                if used_supplement and supplement_df is not None and not supplement_df.empty:
                    max_rows = used_supplement.get("max_rows", 200)
                    evidence_pack = build_dual_source_pack(
                        ccts_df=supplement_df,
                        weekly_blob=data_context,
                        max_rows=max_rows
                    )
                    sources_line = format_sources(used_supplement.get("file"), file_path)
                    if sources_line:
                        evidence_pack += f"\n\n{sources_line}"

                    # Inject style guidance for CCTS answers (inline evidence tags)
                    extra_instruction = ccts_answer_style_instruction()

                    final_context = evidence_pack
                # --------- end Plan C routing ----------

                # Merge extra instruction into question context (so it affects style)
                effective_qctx = (q_ctx + "\n\n" + extra_instruction).strip() if extra_instruction else q_ctx

                response = analyzer.generate_ai_content(
                    unified_prompt=unified_prompt,
                    question=user_input,
                    data_context=final_context,
                    question_context=effective_qctx
                )
                st.write(response)
                st.session_state.conversation.append({"role": "assistant", "content": response})

    elif user_input and df is None:
        with st.chat_message("assistant"):
            st.write("Please ensure the report file is available before asking questions.")

    if st.session_state.conversation:
        st.divider()
        st.subheader("Conversation History")
        conv_text = conversation_to_text(st.session_state.conversation)
        txt_bytes = conv_text.encode("utf-8")
        pdf_bytes = conversation_to_pdf(st.session_state.conversation)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        st.download_button(
            "Download Conversation (TXT)",
            data=txt_bytes,
            file_name=f"conversation_{timestamp}.txt",
            mime="text/plain",
        )
        if pdf_bytes:
            st.download_button(
                "Download Conversation (PDF)",
                data=pdf_bytes,
                file_name=f"conversation_{timestamp}.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Install 'fpdf' package to enable PDF export.")
        email_addr = st.text_input("Recipient email address")
        if st.button("Send Email") and email_addr:
            status = send_conversation_email(
                email_addr,
                st.session_state.conversation,
                pdf_bytes,
                file_path,
            )
            st.info(status)

if __name__ == "__main__":
    main()


