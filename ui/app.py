"""Streamlit development/competition UI.

The UI only talks to FastAPI. It never imports Query Engine internals or the
local database directly, and it never reorders/edits model predictions —
it only displays ranked candidates and exports them (see README "UI is
read-only with respect to model results").
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("AIC_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AIC 2026 — Video Retrieval", layout="wide", page_icon="🎬")

# ---------------------------------------------------------------------------
# Look & feel: dark, grid-based, close to the FiftyOne layout in the brief.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* This is an internal local tool — the Streamlit "Deploy"/menu toolbar
       is not needed and, being fixed-position, was overlapping the top of
       the page content (including the tabs). Hide it outright. */
    header[data-testid="stHeader"] {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}

    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    .aic-card {
        background: #1d2126; border: 1px solid #30353c; border-radius: 10px;
        padding: 10px; margin-bottom: 14px;
    }
    .aic-rank {
        display: inline-block; background: #ff6d00; color: white; font-weight: 700;
        border-radius: 6px; padding: 1px 8px; font-size: 12px; margin-right: 6px;
    }
    .aic-score {color: #9fe870; font-family: monospace; font-size: 12px;}
    .aic-video {font-family: monospace; font-size: 12px; color: #8fb4ff;}
    .aic-answer {
        background: #14251c; border-left: 3px solid #2ecc71; padding: 6px 8px;
        margin-top: 6px; font-size: 13px; border-radius: 4px;
    }
    .aic-event {
        display: inline-block; background: #26303b; border-radius: 6px;
        padding: 2px 6px; margin: 2px; font-size: 11px; font-family: monospace;
    }
    /* Force the tab bar to lay out as a clean single row: some browser/zoom
       combinations otherwise let the two tab labels visually overlap. */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 4px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        white-space: nowrap;
        flex-shrink: 0;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "results" not in st.session_state:
    st.session_state["results"] = {}  # query_id -> SearchResponse dict
if "health" not in st.session_state:
    st.session_state["health"] = None
if "query_id_input" not in st.session_state:
    # Only ever set ONCE, on the very first load. After that, this key is
    # owned entirely by the text_input widget below (via key=), so whatever
    # the user types/presses Enter on is exactly what gets used — it will
    # never silently reset to a fresh random id on rerun.
    st.session_state["query_id_input"] = f"q-{uuid.uuid4().hex[:6]}"


def frame_image_url(video_id: str, frame_id: int) -> str:
    return f"{API_BASE}/api/v1/video/{video_id}/frame/{frame_id}/image"


def run_search(payload: dict) -> None:
    try:
        response = requests.post(f"{API_BASE}/api/v1/search", json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        st.session_state["results"][result["query_id"]] = result
        st.session_state["active_query_id"] = result["query_id"]
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")


# ---------------------------------------------------------------------------
# Sidebar — query builder (KIS / QA / TRAKE)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 AIC 2026")
    st.caption("Truy vấn ảnh / video / âm thanh / nội dung")

    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.success(f"API OK — engine: {health.get('engine')}")
    except requests.RequestException:
        st.error(f"Không kết nối được API tại {API_BASE}")

    st.divider()
    task = st.radio("Chế độ truy vấn", ["KIS", "QA", "TRAKE"], horizontal=True)
    query_id = st.text_input(
        "Query ID",
        key="query_id_input",
        help="Tên này dùng làm tên file .csv khi xuất đáp án. Gõ xong nhấn Enter — giữ nguyên đúng như bạn nhập.",
    )

    if task == "KIS":
        st.caption("Known-Item Search — tìm 1 khoảnh khắc cụ thể.")
        description = st.text_area(
            "Mô tả cảnh cần tìm",
            placeholder="VD: Người đàn ông tóc bạc đang vẫy tay trong studio...",
            height=110,
        )
        if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True):
            if not description.strip():
                st.warning("Nhập mô tả trước khi tìm.")
            else:
                run_search({"query_id": query_id, "task": "KIS", "description": description})

    elif task == "QA":
        st.caption("Q&A — tìm khoảnh khắc rồi trả lời câu hỏi về nó.")
        description = st.text_area("Mô tả cảnh", placeholder="Cảnh liên quan đến câu hỏi...", height=90)
        question = st.text_input("Câu hỏi", placeholder="VD: Người này đang mặc áo màu gì?")
        if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True):
            if not description.strip() or not question.strip():
                st.warning("Nhập cả mô tả và câu hỏi.")
            else:
                run_search(
                    {
                        "query_id": query_id,
                        "task": "QA",
                        "description": description,
                        "question": question,
                    }
                )

    else:  # TRAKE
        st.caption("TRAKE — tìm chuỗi sự kiện theo thứ tự thời gian trong 1 video.")
        n_events = st.number_input("Số sự kiện", min_value=2, max_value=10, value=3, step=1)
        events = []
        for i in range(int(n_events)):
            text = st.text_input(f"Sự kiện E{i + 1}", key=f"event_{i}")
            events.append({"event_id": f"E{i + 1}", "description": text})
        if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True):
            if any(not e["description"].strip() for e in events):
                st.warning("Điền mô tả cho tất cả sự kiện.")
            else:
                run_search({"query_id": query_id, "task": "TRAKE", "events": events})

    st.divider()
    st.caption(f"API: {API_BASE}")


# ---------------------------------------------------------------------------
# Main area — results grid + submission export
# ---------------------------------------------------------------------------
tab_results, tab_submission = st.tabs(["📊 Kết quả truy vấn", "📤 Xuất đáp án bài thi"])

with tab_results:
    results = st.session_state["results"]
    if not results:
        st.info("Chưa có truy vấn nào. Dùng thanh bên trái để bắt đầu (KIS / Q&A / TRAKE).")
    else:
        query_ids = list(results.keys())
        active = st.selectbox(
            "Xem kết quả của query",
            query_ids,
            index=query_ids.index(st.session_state.get("active_query_id", query_ids[-1]))
            if st.session_state.get("active_query_id") in query_ids
            else len(query_ids) - 1,
        )
        result = results[active]

        status = result.get("status")
        if status == "failed":
            st.error(f"Truy vấn thất bại: {result.get('error')}")
        else:
            st.subheader(f"{result['task']} — `{result['query_id']}`")
            st.caption(f"Trạng thái: {status} · {len(result.get('candidates', []))} kết quả (tối đa 100)")

            candidates = result.get("candidates", [])
            n_cols = 4
            cols = st.columns(n_cols)
            for i, cand in enumerate(candidates):
                col = cols[i % n_cols]
                with col:
                    st.markdown('<div class="aic-card">', unsafe_allow_html=True)
                    if result["task"] == "TRAKE":
                        events = cand.get("events", [])
                        if events:
                            first = events[0]
                            st.image(
                                frame_image_url(cand["video_id"], first["frame_id"]),
                                use_container_width=True,
                            )
                        st.markdown(
                            f'<span class="aic-rank">#{cand.get("rank")}</span>'
                            f'<span class="aic-video">{cand["video_id"]}</span> '
                            f'<span class="aic-score">score {cand.get("score", 0):.3f}</span>',
                            unsafe_allow_html=True,
                        )
                        badges = "".join(
                            f'<span class="aic-event">{e["event_id"]}: f{e["frame_id"]}</span>'
                            for e in events
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        if cand.get("frame_id") is not None:
                            st.image(
                                frame_image_url(cand["video_id"], cand["frame_id"]),
                                use_container_width=True,
                            )
                        st.markdown(
                            f'<span class="aic-rank">#{cand.get("rank")}</span>'
                            f'<span class="aic-video">{cand["video_id"]} · frame {cand.get("frame_id")}</span><br>'
                            f'<span class="aic-score">score {cand.get("score", 0):.3f}</span>',
                            unsafe_allow_html=True,
                        )
                        if "answer" in cand:
                            st.markdown(
                                f'<div class="aic-answer">💬 {cand["answer"] or "(chưa có câu trả lời)"}</div>',
                                unsafe_allow_html=True,
                            )
                    with st.expander("Chi tiết / evidence"):
                        st.json(cand.get("evidence", {}))
                    st.markdown("</div>", unsafe_allow_html=True)

with tab_submission:
    st.subheader("Xuất đáp án nộp bài")
    st.caption(
        "Chọn các query đã chạy để đóng gói thành file đáp án (CSV theo từng query, "
        "nén chung 1 file .zip). Danh sách và thứ hạng giữ nguyên như engine trả về — "
        "không chỉnh sửa thủ công."
    )
    results = st.session_state["results"]
    if not results:
        st.info("Chưa có kết quả nào để xuất.")
    else:
        completed = {qid: r for qid, r in results.items() if r.get("status") == "completed"}
        failed = [qid for qid, r in results.items() if r.get("status") != "completed"]
        if failed:
            st.warning(f"Các query sau chưa hoàn tất và sẽ bị bỏ qua: {', '.join(failed)}")

        chosen = st.multiselect(
            "Query đưa vào file đáp án",
            list(completed.keys()),
            default=list(completed.keys()),
        )
        if st.button("📦 Tạo file đáp án", type="primary", disabled=not chosen):
            try:
                response = requests.post(
                    f"{API_BASE}/api/v1/submission",
                    json={"query_ids": chosen},
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") == "completed":
                    st.session_state["submission_file"] = payload["file_name"]
                    st.success(f"Đã tạo: {payload['file_name']}")
                else:
                    st.error(payload.get("error", "Tạo file đáp án thất bại."))
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")

        file_name = st.session_state.get("submission_file")
        if file_name:
            try:
                file_resp = requests.get(f"{API_BASE}/api/v1/submission/{file_name}", timeout=60)
                file_resp.raise_for_status()
                st.download_button(
                    "⬇️ Tải file đáp án (.zip)",
                    data=file_resp.content,
                    file_name=file_name,
                    mime="application/zip",
                )
            except requests.RequestException as exc:
                st.error(f"Không tải được file: {exc}")
