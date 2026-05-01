import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

st.set_page_config(
    page_title="PDF · RAG",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #f7f6f3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #aaaaaa !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #f0ede8 !important;
}
[data-testid="stSidebar"] .sidebar-text {
    color: #888888 !important;
}

/* ── Main heading ── */
.page-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #999999;
    margin-bottom: 10px;
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #111111;
    line-height: 1.1;
    margin-bottom: 6px;
}
.page-rule {
    border: none;
    border-top: 2px solid #111111;
    margin: 18px 0 28px 0;
}
.page-desc {
    font-size: 0.92rem;
    color: #666666;
    font-weight: 300;
    line-height: 1.7;
    max-width: 520px;
    margin-bottom: 32px;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background-color: #ffffff !important;
    border: 1px solid #d8d5cf !important;
    border-radius: 4px !important;
    color: #111111 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #111111 !important;
    box-shadow: none !important;
}
.stTextInput label, .stTextArea label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #888888 !important;
    margin-bottom: 6px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px dashed #cccccc !important;
    border-radius: 6px !important;
    padding: 20px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #888888 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #888888 !important;
    font-size: 0.85rem !important;
}

/* ── Button ── */
.stButton > button {
    background-color: #111111 !important;
    color: #f7f6f3 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    padding: 11px 26px !important;
    transition: background-color 0.2s, opacity 0.2s !important;
}
.stButton > button:hover {
    background-color: #2a2a2a !important;
}
.stButton > button:active {
    opacity: 0.85 !important;
}

/* ── Alerts ── */
.stSuccess > div {
    background-color: #f0f0ec !important;
    border: 1px solid #c8c5be !important;
    border-left: 3px solid #111111 !important;
    border-radius: 4px !important;
    color: #333333 !important;
    font-size: 0.85rem !important;
}
.stInfo > div {
    background-color: #f5f4f1 !important;
    border: 1px solid #d8d5cf !important;
    border-left: 3px solid #aaaaaa !important;
    border-radius: 4px !important;
    color: #555555 !important;
    font-size: 0.85rem !important;
}
.stWarning > div {
    background-color: #f7f5f0 !important;
    border: 1px solid #d8d5cf !important;
    border-left: 3px solid #888888 !important;
    border-radius: 4px !important;
    color: #555555 !important;
    font-size: 0.85rem !important;
}
.stError > div {
    background-color: #f7f4f4 !important;
    border: 1px solid #ddd0d0 !important;
    border-left: 3px solid #555555 !important;
    border-radius: 4px !important;
    color: #444444 !important;
    font-size: 0.85rem !important;
}

/* ── Response card ── */
.response-card {
    background: #ffffff;
    border: 1px solid #d8d5cf;
    border-radius: 6px;
    padding: 28px 32px;
    margin-top: 20px;
}
.response-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #aaaaaa;
    margin-bottom: 14px;
    border-bottom: 1px solid #eeece8;
    padding-bottom: 10px;
}
.response-body {
    font-size: 0.95rem;
    color: #222222;
    line-height: 1.75;
    font-weight: 300;
}

/* ── Section label ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #aaaaaa;
    margin: 28px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    border-top: 1px solid #e0ddd8;
}

/* ── Footer ── */
.footer-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #cccccc;
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid #e0ddd8;
    letter-spacing: 0.08em;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #111111 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d8d5cf !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #e0ddd8 !important; margin: 24px 0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f7f6f3; }
::-webkit-scrollbar-thumb { background: #cccccc; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding: 8px 0 24px 0;">
  <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.22em; text-transform:uppercase; color:#555; margin-bottom:12px;">Acerca de</div>
  <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.9rem; color:#aaa; line-height:1.7; font-weight:300;">
    Este agente usa recuperación aumentada (RAG) para responder preguntas sobre el contenido de tu PDF usando OpenAI.
  </div>
  <div style="margin-top:28px; font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.22em; text-transform:uppercase; color:#555; margin-bottom:12px;">Cómo funciona</div>
  <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem; color:#777; line-height:1.8; font-weight:300;">
    01 — Carga tu PDF<br>
    02 — El documento se fragmenta<br>
    03 — Se crean embeddings<br>
    04 — Haz tu pregunta<br>
    05 — El modelo responde
  </div>
</div>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-eyebrow">Recuperación Aumentada · RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Pregunta sobre<br>tu documento.</div>', unsafe_allow_html=True)
st.markdown('<hr class="page-rule">', unsafe_allow_html=True)
st.markdown('<div class="page-desc">Carga cualquier PDF y haz preguntas en lenguaje natural. El sistema extrae los fragmentos más relevantes y genera una respuesta precisa usando GPT-4.</div>', unsafe_allow_html=True)

# ── Image ──────────────────────────────────────────────────────────────────────
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=320)
    st.markdown("<br>", unsafe_allow_html=True)
except Exception:
    pass

# ── API Key ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Credenciales</div>', unsafe_allow_html=True)
ke = st.text_input('Clave de API de OpenAI', type="password", placeholder="sk-...")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Ingresa tu clave de OpenAI para continuar.")

# ── PDF Upload ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Documento</div>', unsafe_allow_html=True)
pdf = st.file_uploader("Sube tu archivo PDF", type="pdf", label_visibility="collapsed")

# ── Processing ─────────────────────────────────────────────────────────────────
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"Documento cargado — {len(text):,} caracteres extraídos")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Listo — {len(chunks)} fragmentos indexados")

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown('<div class="section-label">Pregunta</div>', unsafe_allow_html=True)
        user_question = st.text_area(
            "Escribe tu pregunta",
            placeholder="¿Qué quieres saber sobre el documento?",
            label_visibility="collapsed",
            height=100
        )

        if user_question:
            with st.spinner("Buscando respuesta..."):
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

            st.markdown(f"""
<div class="response-card">
  <div class="response-label">Respuesta</div>
  <div class="response-body">{response}</div>
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error(f"Error al procesar el PDF: {str(e)}")
        with st.expander("Ver detalle del error"):
            st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Ingresa tu clave de OpenAI para procesar el documento.")
else:
    st.info("Sube un archivo PDF para comenzar.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer-line">Python {platform.python_version()} · RAG con LangChain + OpenAI</div>',
    unsafe_allow_html=True
)
