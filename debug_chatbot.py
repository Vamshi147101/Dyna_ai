# app.py
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
import os
import re
import tempfile

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="LS-DYNA Debug Assistant",
    page_icon="🔧",
    layout="wide"
)

# ─── INIT SESSION STATE ───
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "current_file_content" not in st.session_state:
    st.session_state.current_file_content = None

# ─── CONFIG ───
CHROMA_PATH = "./K-files/Dyna_ai/chroma_db_v2"
GEMINI_MODEL = "google/gemma-4-26b-a4b-it:free"

# ─── LOAD VECTOR DB ───
@st.cache_resource
def load_vector_db():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )
    return {
        "keyword": chroma_client.get_collection("keyword_manual", embedding_function=embedding_fn),
        "material": chroma_client.get_collection("material_manual", embedding_function=embedding_fn),
        "example": chroma_client.get_collection("example_manual", embedding_function=embedding_fn),
    }

collections = load_vector_db()

# ─── RETRIEVAL ───
def retrieve_context(query, n=3):
    parts = []
    for name, coll in collections.items():
        results = coll.query(query_texts=[query], n_results=n)
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            score = 1 - results['distances'][0][i]
            if score > 0.5:
                parts.append(
                    f"[{name}_manual | page {meta.get('page','?')} | "
                    f"card: {meta.get('primary_card','N/A')}]\n{doc}"
                )
    return "\n\n---\n\n".join(parts) if parts else ""

# ─── GEMINI ───
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

SYSTEM_PROMPT = """You are an expert LS-DYNA debugging assistant.
You help users understand and fix their LS-DYNA keyword (.k) files.

You can:
- Explain what any *CARD does and what its parameters mean
- Identify common mistakes in keyword definitions
- Suggest which cards are needed for specific simulation scenarios
- Explain error messages and how to fix them
- Compare material models and element formulations
- Analyze .k file snippets line by line for errors and missing requirements

Rules:
- Be specific. Cite exact field names (MID, RO, E, PR, etc.)
- When referencing documentation, mention card name and field names
- If unsure, say so and suggest what to check
- Keep answers practical and actionable
- When analyzing files, be thorough but concise"""

# ─── ANALYZE UPLOADED FILE ───
def quick_check_k_file(content):
    """Basic checks on .k file content"""
    issues = []
    warnings = []
    
    if '*END' not in content.upper():
        issues.append("❌ Missing *END card - LS-DYNA requires this")
    if '*CONTROL_TERMINATION' not in content.upper():
        issues.append("❌ Missing *CONTROL_TERMINATION - termination time not defined")
    if '*PART' not in content.upper():
        issues.append("❌ Missing *PART card - no parts defined")
    if '*DATABASE_BINARY_D3PLOT' not in content.upper():
        warnings.append("⚠️ No D3PLOT output configured - no visualization files will be generated")
    if '*CONTROL_TIMESTEP' not in content.upper():
        warnings.append("⚠️ No *CONTROL_TIMESTEP - timestep will be auto-calculated")
    
    # Check Poisson's ratio
    pr_matches = re.findall(r'^\s+[\d.]+\s+[\d.+-]+\s+[\d.+-]+\s+([\d.]+)', content, re.MULTILINE)
    for pr in pr_matches:
        try:
            if float(pr) >= 0.5:
                warnings.append(f"⚠️ Poisson's ratio = {pr} may cause volume locking in explicit")
        except ValueError:
            pass
    
    cards = list(set(re.findall(r'\*[A-Z][A-Z_0-9_]+', content)))
    
    return cards, issues, warnings

# ─── SIDEBAR ───
with st.sidebar:
    st.title("🔧 LS-DYNA Debugger")
    st.markdown("---")
    
    # API Key
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/apikey"
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    st.markdown("---")
    
    # File Upload
    st.subheader("📁 Upload .k File")
    uploaded_file = st.file_uploader(
        "Choose a .k file",
        type=['k', 'key', 'dyn', 'txt'],
        help="Upload an LS-DYNA keyword file for analysis"
    )
    
    if uploaded_file:
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        st.session_state.current_file_content = content
        st.session_state.uploaded_files[uploaded_file.name] = content
        
        cards, issues, warnings = quick_check_k_file(content)
        
        st.markdown("---")
        st.subheader(f"📄 {uploaded_file.name}")
        st.metric("Cards Found", len(cards))
        
        if issues:
            st.markdown("**Issues:**")
            for issue in issues:
                st.markdown(issue)
        
        if warnings:
            st.markdown("**Warnings:**")
            for warning in warnings:
                st.markdown(warning)
        
        with st.expander("View Cards"):
            st.markdown(", ".join(sorted(cards)))
        
        if st.button("🔍 Analyze File", use_container_width=True):
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"Analyze this .k file ({uploaded_file.name})"
            })
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_file_content = None
        st.rerun()

# ─── MAIN CHAT AREA ───
st.title("💬 Chat")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about LS-DYNA cards, errors, or best practices..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        client = get_gemini_client()
        
        if not client:
            st.error("Please enter your Gemini API key in the sidebar")
            st.stop()
        
        # Check if there's a file to analyze
        has_file = st.session_state.current_file_content is not None
        file_analysis_requested = any(
            "analyze this .k file" in prompt.lower() or 
            "analyze the file" in prompt.lower()
            for prompt in [msg["content"] for msg in st.session_state.chat_history[-3:]]
        )
        
        # Retrieve context
        with st.spinner("Retrieving documentation..."):
            if has_file and (file_analysis_requested or "file" in prompt.lower()):
                cards, _, _ = quick_check_k_file(st.session_state.current_file_content)
                context = retrieve_context(', '.join(sorted(cards)[:30]))
                
                full_prompt = f"""## Relevant Documentation
{context}
## User Request
{prompt}

Analyze this file and answer the user's question."""
            else:
                context = retrieve_context(prompt)
                full_prompt = f"""## Relevant Documentation
{context}

## User Question
{prompt}"""
        
        # Build conversation for Gemini
        contents = []
        for turn in st.session_state.chat_history[-8:]:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})
        
        # Add current prompt with context
        contents.append({"role": "user", "parts": [{"text": full_prompt}]})
        
        with st.spinner("Thinking..."):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": 0.3,
                        "max_output_tokens": 4096,
                    }
                )
                reply = response.text
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"Error: {e}")

# ─── FOOTER ───
st.markdown("---")
st.caption("LS-DYNA® is a registered trademark of ANSYS® Inc.")