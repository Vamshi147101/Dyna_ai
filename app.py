# app.py
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
import os
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ─── PAGE CONFIGURATION ───
st.set_page_config(
    page_title="LS-DYNA Debug Assistant",
    page_icon="🔧",
    layout="wide"
)

# ─── SESSION STATE ───
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "file_cards" not in st.session_state:
    st.session_state.file_cards = []
if "file_issues" not in st.session_state:
    st.session_state.file_issues = []

# ─── CONFIG ───
CHROMA_PATH = "./K-files/Dyna_ai/chroma_db_v2"
GEMINI_MODEL = "gemini-2.5-flash"

# ─── LOAD DB ONCE ───
@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )
    return {
        "keyword": client.get_collection("keyword_manual", embedding_function=ef),
        "material": client.get_collection("material_manual", embedding_function=ef),
        "example": client.get_collection("example_manual", embedding_function=ef),
    }

db = load_db()

# ─── PARALLEL RETRIEVAL ───
def query_collection(name, coll, query, n):
    results = coll.query(query_texts=[query], n_results=n)
    parts = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        score = 1 - results['distances'][0][i]
        if score > 0.5:
            parts.append(
                f"[{name} | p{meta.get('page','?')} | "
                f"{meta.get('primary_card','')}]\n{doc[:800]}"  # Truncate chunks
            )
    return parts

def retrieve_context(query, n=2):  # Reduced from 3 to 2
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(query_collection, name, coll, query, n): name
            for name, coll in db.items()
        }
        all_parts = []
        for future in futures:
            all_parts.extend(future.result())
    
    return "\n\n---\n\n".join(all_parts[:5])  # Max 5 chunks total

# ─── QUICK FILE CHECK ───
def check_file(content):
    issues = []
    warnings = []
    
    if '*END' not in content.upper():
        issues.append("❌ Missing *END card")
    if '*CONTROL_TERMINATION' not in content.upper():
        issues.append("❌ Missing *CONTROL_TERMINATION")
    if '*PART' not in content.upper():
        issues.append("❌ Missing *PART card")
    if '*DATABASE_BINARY_D3PLOT' not in content.upper():
        warnings.append("⚠️ No D3PLOT output")
    
    cards = list(set(re.findall(r'\*[A-Z][A-Z_0-9_]+', content)))
    return cards, issues, warnings

# ─── GEMINI ───
SYSTEM_PROMPT = """You are an LS-DYNA expert. Answer concisely using the documentation provided.
Cite specific card names and field names (MID, RO, E, PR, etc.).
If unsure, say so. Keep responses practical and under 500 words unless analyzing a file."""

def get_client():
    key = os.environ.get("GEMINI_API_KEY") or st.session_state.get("api_key", "")
    return genai.Client(api_key=key) if key else None

def call_gemini(client, contents):
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.2,
            "max_output_tokens": 2048,  # Reduced from 4096
        }
    )

# ─── SIDEBAR ───
with st.sidebar:
    st.title("🔧 LS-DYNA Debugger")
    
    # API Key
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        key="api_key",
        help="https://aistudio.google.com/apikey"
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    st.divider()
    
    # File Upload
    uploaded = st.file_uploader("Upload .k file", type=['k', 'key', 'dyn', 'txt'])
    
    if uploaded:
        content = uploaded.getvalue().decode('utf-8', errors='ignore')
        
        # Only process if file changed
        if st.session_state.file_name != uploaded.name:
            st.session_state.file_content = content
            st.session_state.file_name = uploaded.name
            st.session_state.file_cards, st.session_state.file_issues, warnings = check_file(content)
            
            # Store warnings in session state too
            st.session_state.file_warnings = warnings
        
        st.metric("Cards", len(st.session_state.file_cards))
        
        for issue in st.session_state.file_issues:
            st.markdown(issue)
        for warning in st.session_state.file_warnings:
            st.markdown(warning)
        
        with st.expander("Card List"):
            st.text(", ".join(sorted(st.session_state.file_cards)[:50]))
        
        if st.button("🔍 Analyze File", use_container_width=True):
            st.session_state.chat_history = []  # Clear for new analysis
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.file_content = None
        st.session_state.file_name = None
        st.rerun()

# ─── MAIN ───
st.title("💬 LS-DYNA Debug Chat")

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask anything about LS-DYNA..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        client = get_client()
        if not client:
            st.error("Enter your Gemini API key in the sidebar")
            st.stop()
        
        # Build prompt
        context = ""
        if st.session_state.file_content and st.session_state.chat_history:
            first_msg = st.session_state.chat_history[0]["content"]
            if "analyze" in first_msg.lower() or "file" in prompt.lower():
                # File analysis mode - limit content
                cards_str = ', '.join(sorted(st.session_state.file_cards)[:20])
                file_snippet = st.session_state.file_content[:5000]  # Reduced from 15000
                context = retrieve_context(cards_str)
                
                full_prompt = f"""## Docs
{context[:2000]}

## File Cards: {cards_str}

## File:
{file_snippet}

## Question: {prompt}"""
            else:
                context = retrieve_context(prompt)
                full_prompt = f"## Docs\n{context[:1500]}\n\n## Question\n{prompt}"
        else:
            context = retrieve_context(prompt)
            full_prompt = f"## Docs\n{context[:1500]}\n\n## Question\n{prompt}"
        
        # Build contents (last 4 turns only)
        contents = []
        for turn in st.session_state.chat_history[-6:]:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": turn["content"][:1000]}]})
        contents.append({"role": "user", "parts": [{"text": full_prompt[:4000]}]})
        
        with st.spinner("..."):
            try:
                response = call_gemini(client, contents)
                reply = response.text
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")