"""
TruthLens AI — Streamlit frontend.

Run locally:
    streamlit run frontend/app.py
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="TruthLens AI", layout="wide")

st.sidebar.title("TruthLens AI")
page = st.sidebar.radio(
    "Navigate",
    ["Chat", "Upload", "Evidence Viewer", "Trust Dashboard", "Sources", "Evaluation"],
)

if page == "Chat":
    st.title("Chat")
    question = st.text_input("Ask a question")
    if st.button("Ask") and question:
        try:
            resp = requests.post(f"{BACKEND_URL}/ask", json={"question": question}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            st.write(data["answer"])
            if data.get("trust_score") is not None:
                st.metric("Trust Score", f"{data['trust_score']*100:.0f}%")
        except Exception as e:
            st.error(f"Backend not reachable yet: {e}")

elif page == "Upload":
    st.title("Upload Documents")
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    if uploaded:
        st.info("Ingestion pipeline not yet wired (Phase 3).")

else:
    st.title(page)
    st.info("This page will be built in a later phase.")
