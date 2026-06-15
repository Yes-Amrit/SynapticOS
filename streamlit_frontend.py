import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    get_all_chat_titles,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_thread(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    try:
        st.session_state["thread_titles"] = get_all_chat_titles()
    except Exception:
        st.session_state["thread_titles"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# Refresh titles dynamically
try:
    st.session_state["thread_titles"] = get_all_chat_titles()
except Exception:
    pass

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("SynapticOS")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# --- PDF Upload Section ---
st.sidebar.subheader("Document for this chat")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
            st.rerun()

# --- Chat History Section ---
st.sidebar.header("Chat History")

for t_id in reversed(st.session_state["chat_threads"]):
    title = st.session_state["thread_titles"].get(t_id, f"Chat {t_id[:8]}")

    if st.sidebar.button(title, key=f"thread_{t_id}", use_container_width=True):
        st.session_state["thread_id"] = t_id
        messages = load_conversation(t_id)
        temp_messages = []

        for msg in messages:
            # Skip rendering background structural tools or system templates
            if isinstance(msg, ToolMessage) or isinstance(msg, SystemMessage):
                continue
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            if msg.content:
                temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages
        st.session_state["ingested_docs"].setdefault(str(t_id), {})
        st.rerun()


# ============================================================================
# MAIN CHAT WINDOW
# ============================================================================
st.title("Multi Utility RAG Chatbot")

# 1. Render Chat History
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

current_thread = st.session_state["thread_id"]
CONFIG = {
    "configurable": {"thread_id": current_thread},
    "metadata": {"thread_id": current_thread},
    "run_name": "chat_turn",
}

# 2. Human-in-the-Loop Interception
state = chatbot.get_state(CONFIG)
is_paused = bool(state.next and "tools" in state.next)

if is_paused:
    last_msg = state.values["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    
    if tool_calls:
        with st.chat_message("assistant"):
            st.warning("✋ **Action Required:** The AI wants to execute the following tool operations:")
            for tc in tool_calls:
                st.markdown(f"* **Tool Name:** `{tc['name']}`")
                st.json(tc["args"])
            
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve Action", use_container_width=True):
                st.session_state["resume_graph"] = True
                st.rerun()
                
            if col2.button("❌ Reject Action", use_container_width=True):
                # A) Add a clear system/assistant note directly to the visible chat history
                st.session_state["message_history"].append({
                    "role": "assistant", 
                    "content": "❌ **Tool execution rejected by user.**"
                })
                
                # B) Inform the graph backend that the action was rejected
                rejection_msgs = [
                    ToolMessage(
                        content="User denied permission to run this tool. Do not try again. Let the user know you understand it was rejected and ask how to proceed.", 
                        name=tc["name"], 
                        tool_call_id=tc["id"]
                    ) for tc in tool_calls
                ]
                chatbot.update_state(CONFIG, {"messages": rejection_msgs}, as_node="tools")
                
                # C) Force the graph to resume running so the AI reads this rejection and responds!
                st.session_state["resume_graph"] = True
                st.rerun()

# 3. Handle Inputs (Disable box if currently waiting on confirmation)
user_input = st.chat_input("Ask about your document or use tools", disabled=is_paused)

if user_input or st.session_state.get("resume_graph"):
    
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        stream_input = {"messages": [HumanMessage(content=user_input)]}
    else:
        # Resuming from a paused state (either via Approval OR Rejection) requires passing None
        stream_input = None 
        st.session_state["resume_graph"] = False

    with st.chat_message("assistant"):
        status_box = None
        message_placeholder = st.empty()
        full_response = ""

        try:
            for chunk, metadata in chatbot.stream(
                stream_input,
                config=CONFIG,
                stream_mode="messages",
            ):
                # Monitor streaming nodes 
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_box is None:
                        status_box = st.status(f"🔧 Running `{tool_name}` …", expanded=True)
                    else:
                        status_box.update(label=f"🔧 Running `{tool_name}` …", state="running")

                elif isinstance(chunk, AIMessage):
                    if chunk.content:
                        if isinstance(chunk.content, str):
                            full_response += chunk.content
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, dict) and "text" in part:
                                    full_response += part["text"]
                                elif isinstance(part, str):
                                    full_response += part

                        message_placeholder.markdown(full_response + " ▌")

            if full_response.strip():
                message_placeholder.markdown(full_response)
                st.session_state["message_history"].append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.markdown("✅ Processing Complete.")

            if status_box is not None:
                status_box.update(label="✅ Tool finished", state="complete", expanded=False)

        except Exception as e:
            message_placeholder.markdown(f"⚠️ An error occurred: {str(e)}")

    st.rerun()

doc_meta = thread_document_metadata(current_thread)
if doc_meta:
    st.caption(
        f"Document indexed: {doc_meta.get('filename')} "
        f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
    )