# app.py
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="LS-DYNA Debug Assistant - Smart Memory",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS - DARK MODE, NO WHITE ───
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%) !important; }
    .main { background: transparent !important; }
    .stMarkdown, .stMarkdown p, div, p, span, li, label, .stTextInput, .stSelectbox { color: #e0e0e0 !important; }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #ffffff !important; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a15 0%, #0f0f1e 100%) !important;
        border-right: 1px solid rgba(102,126,234,0.3) !important;
    }
    
    .stChatMessage { margin-bottom: 16px; border-radius: 12px; overflow: hidden; animation: fadeIn 0.3s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    
    [data-testid="stChatMessage"]:has([data-testid="stMarkdown"]:first-child) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 12px 16px !important; border-radius: 12px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stMarkdown"]:last-child) {
        background: #1e1e2e !important; border-left: 4px solid #667eea !important;
        border-radius: 12px !important; padding: 16px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    
    pre { background: #0d0d1a !important; border-radius: 8px !important; padding: 12px !important; border: 1px solid #2d2d42 !important; }
    code { background: #0d0d1a !important; padding: 2px 6px !important; border-radius: 4px !important; color: #a78bfa !important; }
    pre code { color: #e0e0e0 !important; background: transparent !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        padding: 8px 16px !important; font-weight: 500 !important; transition: all 0.3s !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(102,126,234,0.4) !important; }
    
    .welcome-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        border-radius: 16px !important; padding: 32px !important; text-align: center !important;
        margin-bottom: 24px !important; border: 1px solid rgba(102,126,234,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "file_content" not in st.session_state: st.session_state.file_content = None
if "file_name" not in st.session_state: st.session_state.file_name = None
if "file_cards" not in st.session_state: st.session_state.file_cards = []
if "file_issues" not in st.session_state: st.session_state.file_issues = []
if "file_warnings" not in st.session_state: st.session_state.file_warnings = []
if "file_analyzed" not in st.session_state: st.session_state.file_analyzed = False
if "selected_model" not in st.session_state: st.session_state.selected_model = "gemini-2.5-flash"
if "welcome_shown" not in st.session_state: st.session_state.welcome_shown = False
if "conversation_memory" not in st.session_state: st.session_state.conversation_memory = []
if "api_key_set" not in st.session_state: st.session_state.api_key_set = False
if "gemini_api_key" not in st.session_state: st.session_state.gemini_api_key = ""

# ─── CONFIG ───
CHROMA_PATH = "./K-files/Dyna_ai/chroma_db_v2"

GEMINI_MODELS = {
    "Gemini 2.5 Flash-Lite (⚡ Fastest)": "gemini-2.5-flash-lite",
    "Gemini 2.5 Flash (⚖️ Balanced)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (🚀 Powerful)": "gemini-2.5-pro",
}

# ─── CUSTOM MEMORY CLASS ───
class ConversationMemory:
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history:
            self.messages.pop(0)
    
    def get_context(self, n_last=5):
        return self.messages[-n_last:] if self.messages else []
    
    def clear(self):
        self.messages = []
    
    def extract_key_cards(self):
        cards = set()
        for msg in self.messages:
            found = re.findall(r'\*[A-Z_]+', msg.get("content", ""))
            cards.update(found)
        return list(cards)[:10]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_history=10)

# ─── LOAD DB ───
@st.cache_resource
def load_db():
    with st.spinner("📚 Loading LS-DYNA documentation..."):
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-base-en-v1.5"
            )
            collection_names = ["keyword_manual", "material_manual", "example_manual", "theory_manual", "multiphysics_manual"]
            collections = {}
            for name in collection_names:
                try:
                    collections[name] = client.get_collection(name, embedding_function=ef)
                except Exception as e:
                    pass
            return collections
        except Exception as e:
            return {}

db = load_db()

def query_collection(name, coll, query, n):
    try:
        results = coll.query(query_texts=[query], n_results=n)
        parts = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            score = 1 - results['distances'][0][i]
            if score > 0.5:
                parts.append({
                    "source": name, "page": meta.get('page', '?'),
                    "card": meta.get('primary_card', ''), "content": doc[:800], "score": score
                })
        return parts
    except Exception:
        return []

def retrieve_context_with_memory(query, memory, n=2):
    is_followup = False
    followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'then', 'also', 'so']
    query_lower = query.lower()
    
    if any(query_lower.startswith(indicator) or f" {indicator} " in query_lower for indicator in followup_indicators):
        is_followup = True
    elif len(query.split()) < 8:
        is_followup = True
    
    enhanced_query = query
    if is_followup and memory.messages:
        for msg in reversed(memory.messages):
            if msg["role"] == "assistant":
                cards = re.findall(r'\*[A-Z_]+', msg["content"])
                if cards:
                    enhanced_query = f"{query} related to {' '.join(cards[:3])}"
                    break
    
    if db:
        with ThreadPoolExecutor(max_workers=len(db)) as executor:
            futures = {executor.submit(query_collection, name, coll, enhanced_query, n): name for name, coll in db.items()}
            all_parts = []
            for future in futures:
                all_parts.extend(future.result())
        all_parts.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_parts[:5], is_followup
    return [], is_followup

def check_file(content):
    issues = []
    warnings = []
    if '*END' not in content.upper(): issues.append("Missing *END card")
    if '*CONTROL_TERMINATION' not in content.upper(): issues.append("Missing *CONTROL_TERMINATION")
    if '*PART' not in content.upper(): issues.append("Missing *PART card")
    if '*DATABASE_BINARY_D3PLOT' not in content.upper(): warnings.append("No D3PLOT output configured")
    if '*CONTROL_TIMESTEP' not in content.upper(): warnings.append("No *CONTROL_TIMESTEP - using default")
    cards = list(set(re.findall(r'\*[A-Z][A-Z_0-9_]+', content)))
    return cards, issues, warnings

# ─── AGENT TOOLS ───
@tool
def execute_lsdyna_code(python_code: str) -> str:
    """
    Executes Python code directly in the LS-DYNA / LS-PrePost environment.
    Use this to manipulate the model, create parts, or change settings using the lspost module.
    
    Args:
        python_code (str): Valid Python code using the LS-DYNA/lspost Python API.
    """
    try:
        response = requests.post(
            "http://127.0.0.1:5000", 
            json={"code": python_code},
            timeout=10
        )
        result = response.json()
        
        if result.get("status") == "success":
            return "Execution successful."
        else:
            return f"Error executing code in LS-DYNA: {result.get('message')}"
            
    except requests.exceptions.ConnectionError:
        return "Connection failed. Please ensure the listener script is running in the LS-PrePost console."
    except Exception as e:
        return f"Request failed: {str(e)}"

# ─── LANGGRAPH STATE ───
class ConversationState(TypedDict):
    user_query: str
    enhanced_query: str
    chat_history: List[Dict]
    conversation_memory: List[Dict]
    file_context: Dict
    retrieved_docs: List[Dict]
    analysis_result: str
    is_followup: bool
    iteration: int

# ─── LANGGRAPH NODES ───
def analyze_query(state: ConversationState) -> ConversationState:
    query = state["user_query"]
    followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'then', 'also', 'so']
    query_lower = query.lower()
    is_followup = any(query_lower.startswith(indicator) or f" {indicator} " in query_lower for indicator in followup_indicators)
    if not is_followup and len(query.split()) < 8:
        is_followup = True
    state["is_followup"] = is_followup
    
    enhanced = query
    if is_followup and state["conversation_memory"]:
        for msg in reversed(state["conversation_memory"]):
            if msg.get("role") == "assistant":
                cards = re.findall(r'\*[A-Z_]+', msg.get("content", ""))
                if cards:
                    enhanced = f"{query} (continuing about {' '.join(cards[:2])})"
                    break
    state["enhanced_query"] = enhanced
    return state

def retrieve_documents(state: ConversationState) -> ConversationState:
    docs, _ = retrieve_context_with_memory(state["enhanced_query"], st.session_state.memory, n=3)
    state["retrieved_docs"] = docs
    return state

def generate_response(state: ConversationState) -> ConversationState:
    api_key = st.session_state.gemini_api_key
    if not api_key:
        state["analysis_result"] = "⚠️ Please enter your Gemini API key in the sidebar to continue."
        return state
    
    # Init LLM with tools
    llm = ChatGoogleGenerativeAI(
        model=st.session_state.selected_model,
        google_api_key=api_key,
        temperature=0.1,  # Lower temp is better for code generation
        max_output_tokens=4096,
    )
    llm_with_tools = llm.bind_tools([execute_lsdyna_code])
    
    docs_text = "".join([f"\n**Source {i+1}:** {d['source']} - {d['card']}\n{d['content']}\n" for i, d in enumerate(state.get("retrieved_docs", [])[:4])])
    
    memory_text = ""
    if state["conversation_memory"]:
        last_msgs = state["conversation_memory"][-4:]
        memory_text = "Previous conversation:\n" + "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}" for m in last_msgs])
    
    file_text = ""
    if state["file_context"].get("content"):
        file_text = f"File: {state['file_context'].get('name', 'Unknown')}\nCards: {', '.join(state['file_context'].get('cards', [])[:20])}"
    
    system_prompt = f"""You are an expert LS-DYNA debugging and control assistant.
You have access to a tool that can execute Python code directly in the user's LS-PrePost environment.
If the user asks you to perform an action (e.g., "create a part", "change the timestep"), 
write the appropriate LS-PrePost Python API code (using the `lspost` module) and use the execute_lsdyna_code tool.

{memory_text}
File Context: {file_text}
Documentation: {docs_text}

Answer naturally while maintaining conversation flow. Be specific and actionable."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_query"])
    ]
    
    try:
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM decided to use the tool
        if response.tool_calls:
            tool_msg = ""
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'execute_lsdyna_code':
                    code_to_run = tool_call['args']['python_code']
                    
                    # Execute the tool
                    execution_result = execute_lsdyna_code.invoke({"python_code": code_to_run})
                    
                    tool_msg += f"**Action Executed in LS-PrePost:**\n```python\n{code_to_run}\n```\n"
                    tool_msg += f"**Status:** {execution_result}\n\n"
            
            # Ask the LLM to summarize the result
            followup_messages = messages + [response, HumanMessage(content=f"Tool execution results:\n{tool_msg}\nBriefly tell me what was done.")]
            final_response = llm.invoke(followup_messages)
            
            state["analysis_result"] = tool_msg + final_response.content
        else:
            state["analysis_result"] = response.content
            
    except Exception as e:
        state["analysis_result"] = f"Error: {str(e)}"
    
    return state

def should_continue(state: ConversationState) -> Literal["retrieve_documents", "end"]:
    if state["iteration"] < 1 and len(state.get("analysis_result", "")) < 100:
        state["iteration"] += 1
        return "retrieve_documents"
    return "end"

def create_workflow():
    workflow = StateGraph(ConversationState)
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_response", generate_response)
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_conditional_edges("generate_response", should_continue, {
        "retrieve_documents": "retrieve_documents", "end": END
    })
    return workflow.compile()

# ─── HELPER FUNCTIONS ───
def show_welcome():
    if not st.session_state.welcome_shown and len(st.session_state.chat_history) == 0:
        welcome_msg = """# 👋 Welcome to DYNA-Bot with Smart Memory & Live Control!

### Your Agentic LS-DYNA Assistant!

**Features:**
- 🧠 **Perfect memory** - Remembers our conversation
- ⚙️ **Live Control** - Can execute Python scripts directly in LS-PrePost
- 💡 **Context-aware** - Knows what we discussed
- 🎨 **Dark theme** - Easy on the eyes

**To get started:**
1. Start the listener script inside LS-PrePost Python console.
2. Enter your Gemini API key in the sidebar.
3. Try saying: *"Create a new part named 'Front_Bumper' with Part ID 5"*

Let's build and debug LS-DYNA together! 🚀
"""
        with st.chat_message("assistant"):
            st.markdown(welcome_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
            st.session_state.memory.add_message("assistant", welcome_msg)
            st.session_state.welcome_shown = True

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 48px;">🔧</div>
        <h2 style="margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            DYNA-Bot
        </h2>
        <p style="color: #9ca3af; font-size: 0.85rem;">Agentic Control Mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🔑 Gemini API Key")
    api_key_input = st.text_input("Enter your API key", type="password", key="api_key_input")
    if api_key_input:
        st.session_state.gemini_api_key = api_key_input
        st.session_state.api_key_set = True
        st.success("✅ API key configured!")
    
    st.divider()
    
    st.subheader("🤖 Model")
    selected_display = st.selectbox("Choose Model", options=list(GEMINI_MODELS.keys()), index=1, disabled=not st.session_state.api_key_set)
    st.session_state.selected_model = GEMINI_MODELS[selected_display]

    # ─── RESTORED FILE UPLOAD & DIAGNOSIS UI ───
    st.subheader("📁 Upload .k File")
    uploaded = st.file_uploader(
        "Choose a file",
        type=['k', 'key', 'dyn', 'txt'],
        label_visibility="collapsed",
        disabled=not st.session_state.api_key_set
    )
    
    if uploaded:
        content = uploaded.getvalue().decode('utf-8', errors='ignore')
        
        if st.session_state.file_name != uploaded.name:
            st.session_state.file_content = content
            st.session_state.file_name = uploaded.name
            st.session_state.file_cards, st.session_state.file_issues, warnings = check_file(content)
            st.session_state.file_warnings = warnings
            st.session_state.file_analyzed = False
            st.rerun()
        
        st.markdown(f"""
        <div class="file-info-card">
            <div>📄 {st.session_state.file_name}</div>
            <div style="font-size: 0.85rem;">🏷️ {len(st.session_state.file_cards)} cards found</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.file_issues:
            for issue in st.session_state.file_issues:
                st.markdown(f"<span class='badge badge-error'>ERROR</span> {issue}", unsafe_allow_html=True)
        
        if st.session_state.file_warnings:
            for warning in st.session_state.file_warnings:
                st.markdown(f"<span class='badge badge-warning'>WARN</span> {warning}", unsafe_allow_html=True)
        
        if not st.session_state.file_issues and not st.session_state.file_warnings:
            st.markdown("<span class='badge badge-success'>✓ Valid</span> No critical issues", unsafe_allow_html=True)
        
        with st.expander(f"📋 Cards ({len(st.session_state.file_cards)})"):
            cards_cols = st.columns(2)
            for i, card in enumerate(sorted(st.session_state.file_cards)[:50]):
                with cards_cols[i % 2]:
                    st.markdown(f"`{card}`")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.welcome_shown = False
        st.session_state.memory.clear()
        st.rerun()

# ─── MAIN CHAT ───
st.markdown("""
<div class='welcome-card'>
    <div style="font-size: 48px; margin-bottom: 16px;">💬</div>
    <h1 style="margin: 0;">DYNA-Bot Agent</h1>
    <p style="font-size: 1.1rem; opacity: 0.95;">I can now control LS-PrePost directly.</p>
</div>
""", unsafe_allow_html=True)

show_welcome()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

chat_disabled = not st.session_state.api_key_set

if prompt := st.chat_input("Ask about LS-DYNA cards, or tell me to perform an action...", disabled=chat_disabled):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.memory.add_message("user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if not st.session_state.api_key_set:
            st.error("⚠️ Please enter your Gemini API key in the sidebar first")
            st.stop()
        
        try:
            workflow = create_workflow()
            initial_state = ConversationState(
                user_query=prompt,
                enhanced_query=prompt,
                chat_history=st.session_state.chat_history,
                conversation_memory=st.session_state.memory.messages,
                file_context={"content": st.session_state.file_content, "name": st.session_state.file_name, "cards": st.session_state.file_cards},
                retrieved_docs=[], analysis_result="", is_followup=False, iteration=0
            )
            
            with st.spinner("🤖 Thinking and acting..."):
                final_state = workflow.invoke(initial_state)
            
            if final_state and final_state.get("analysis_result"):
                response = final_state["analysis_result"]
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.memory.add_message("assistant", response)
            else:
                st.error("Failed to generate response")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("""<hr><div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 20px;">🤖 LangGraph Agent • Connects to LS-PrePost locally</div>""", unsafe_allow_html=True)