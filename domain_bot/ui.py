import os, requests, streamlit as st

st.set_page_config(page_title="Domain-Specific Chatbot", page_icon="🧠")
st.title("🧠 Domain-Specific Chatbot (Local LLM + RAG)")

api_url = os.environ.get("API_URL", "http://localhost:8000/chat")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Settings")
    st.text_input("API URL", value=api_url, key="api_url")
    st.button("Clear history", on_click=lambda: st.session_state.update(history=[]))

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_msg = st.chat_input("Ask your question...")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    try:
        resp = requests.post(st.session_state.api_url, json={"query": user_msg, "history": st.session_state.history})
        data = resp.json()
        answer = data.get("answer", "")
    except Exception as e:
        answer = f"Error contacting API: {e}"
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
