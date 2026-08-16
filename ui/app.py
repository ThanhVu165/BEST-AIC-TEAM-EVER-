"""Streamlit development/competition UI.

The UI only talks to FastAPI. It never imports Query Engine internals or the
local database directly.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("AIC_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AIC 2026 Video Retrieval", layout="wide")
st.title("AIC 2026 — Video Retrieval")
st.caption("UI → FastAPI → Query Engine. No manual result editing.")

query_id = st.text_input("Query ID", value="demo-001")
task = st.selectbox("Task", ["KIS", "QA", "TRAKE"])
raw_text = st.text_area("Natural-language query", value="Find the described event in the video.")
question = st.text_input("Question (QA only)") if task == "QA" else ""

if st.button("Search", type="primary"):
    payload = {
        "query_id": query_id,
        "task": task,
        "description": raw_text,
        "question": question or None,
        "raw_text": raw_text,
    }
    try:
        response = requests.post(f"{API_BASE}/api/v1/search", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        st.session_state["last_result"] = result
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")

result = st.session_state.get("last_result")
if result:
    st.subheader(f"Results — {result['task']}")
    for candidate in result.get("candidates", []):
        with st.container(border=True):
            st.markdown(f"**Rank {candidate.get('rank', '?')}**")
            st.write(f"Video: `{candidate.get('video_id')}`")
            if candidate.get("frame_id") is not None:
                st.write(f"Frame: `{candidate['frame_id']}`")
            st.write(f"Score: `{candidate.get('score', 0):.4f}`")
            if "answer" in candidate:
                st.write(f"Answer: {candidate['answer']}")
            if "events" in candidate:
                st.json(candidate["events"])

st.divider()
st.caption(f"API: {API_BASE}")
