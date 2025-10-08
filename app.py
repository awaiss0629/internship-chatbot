# app.py — Internship Support Chatbot (Streamlit + Chroma RAG)
# ------------------------------------------------------------
# - Put this file next to faqs.csv (and optional tickets.csv)
# - Launch in Colab with:  streamlit run app.py --server.port=8501
# - Tunnel via ngrok/Cloudflare/LocalTunnel as you prefer.

import os
import time
import hashlib
import pandas as pd
import streamlit as st
from typing import Tuple, List, Dict, Any

# Vector DB + Embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------------------- Config -----------------------
DATA_FAQS = os.getenv("FAQS_PATH", "faqs.csv")
DATA_TICKETS = os.getenv("TICKETS_PATH", "tickets.csv")  # optional
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "intern_support")
TOP_K = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.45"))  # 0..1, higher = stricter
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "./logs/feedback.csv")

st.set_page_config(page_title="Internship Support Bot", page_icon="🤖", layout="wide")

# ----------------------- Helpers -----------------------
def _dataset_fingerprint(df: pd.DataFrame) -> str:
    cols = [c for c in ["question", "answer", "category", "updated_at"] if c in df.columns]
    s = "|".join(df[cols].astype(str).agg("||".join, axis=1)) if cols else ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))

def clear_collection_safely(client: chromadb.PersistentClient, name: str) -> chromadb.api.models.Collection.Collection:
    """Drop & recreate collection (avoids Chroma's where={} delete restriction)."""
    try:
        client.delete_collection(name)
    except Exception:
        # If it doesn't exist or fails, ignore and continue
        pass
    return client.create_collection(name)

def ensure_collection_up_to_date(
    client: chromadb.PersistentClient,
    embedder: SentenceTransformer,
    faqs_df: pd.DataFrame,
) -> Tuple[chromadb.api.models.Collection.Collection, bool]:
    """
    Returns (collection, rebuilt_flag).
    Rebuilds if fingerprint changed.
    """
    try:
        col = client.get_or_create_collection(COLLECTION)
    except Exception:
        col = client.create_collection(COLLECTION)

    current_fp = _dataset_fingerprint(faqs_df)

    # Check if existing collection matches fingerprint
    needs_build = True
    try:
        meta = col.get(include=["metadatas"], limit=1)
        if meta and meta.get("metadatas"):
            if (meta["metadatas"][0] or {}).get("_fingerprint") == current_fp:
                needs_build = False
    except Exception:
        needs_build = True

    if not needs_build:
        return col, False

    # Drop & recreate to clear
    col = clear_collection_safely(client, COLLECTION)

    # Build embeddings (embed questions; store answers as retrievable documents)
    ids, docs, metas = [], [], []
    for i, r in faqs_df.iterrows():
        q = str(r["question"])
        a = str(r["answer"])
        cat = str(r.get("category", ""))
        _id = str(r.get("id", f"faq_{i}"))
        ids.append(_id)
        docs.append(a)
        metas.append({"question": q, "category": cat, "_fingerprint": current_fp})

    q_embs = embedder.encode(faqs_df["question"].astype(str).tolist(), batch_size=64, show_progress_bar=False)
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=[e.tolist() for e in q_embs])

    return col, True

def answer_from_hits(hits: List[Tuple[str, Dict[str, Any], float]]) -> Tuple[str, List[Tuple[str, Dict[str, Any], float]]]:
    """Return final answer text + the ranked hits for transparency."""
    if not hits:
        return "Sorry, I don’t have an answer for that yet.", []
    best_doc, _, best_dist = hits[0]
    best_sim = 1 - float(best_dist) if isinstance(best_dist, (int, float)) else 0.0
    if best_sim < SIM_THRESHOLD:
        return "I couldn’t confidently find a match. Try rephrasing or pick a category in the sidebar.", hits
    return best_doc, hits

def ensure_feedback_file():
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    if not os.path.exists(FEEDBACK_PATH):
        pd.DataFrame(columns=["ts","question","answer","helpful","notes"]).to_csv(FEEDBACK_PATH, index=False)

def log_feedback(question: str, answer: str, helpful: Any, notes: str = ""):
    ensure_feedback_file()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame([[ts, question, answer, helpful, notes]],
                 columns=["ts","question","answer","helpful","notes"]).to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)

# ----------------------- Sidebar: Data & Filters -----------------------
st.sidebar.title("Settings")

up_faq = st.sidebar.file_uploader("Upload faqs.csv (optional)", type=["csv"])
if up_faq is not None:
    faqs_df = pd.read_csv(up_faq)
else:
    if not os.path.exists(DATA_FAQS):
        st.sidebar.error("faqs.csv not found. Upload here or place it next to app.py.")
        st.stop()
    faqs_df = pd.read_csv(DATA_FAQS)

# Optional tickets (not required for app)
if os.path.exists(DATA_TICKETS):
    tickets_df = pd.read_csv(DATA_TICKETS)
else:
    tickets_df = pd.DataFrame(columns=["ticket_id","user_text","resolved_answer","intent","entities"])

st.sidebar.caption(f"FAQs loaded: **{len(faqs_df)}**")
cat_options = ["All"] + sorted(list(faqs_df.get("category", pd.Series(dtype=str)).dropna().unique()))
cat_filter = st.sidebar.selectbox("Category hint (affects retrieval)", cat_options, index=0)

# ----------------------- Init Vector DB -----------------------
with st.spinner("Loading model & index…"):
    embedder = get_embedder()
    chroma_client = get_chroma_client()
    collection, rebuilt = ensure_collection_up_to_date(chroma_client, embedder, faqs_df)

st.sidebar.success("Index rebuilt from FAQs ✅") if rebuilt else st.sidebar.info("Index loaded ✅")

# ----------------------- UI -----------------------
st.title("🤖 Internship Support Chatbot")
st.caption("Answers grounded in your FAQ data. Shows sources and collects feedback.")

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.expander("Suggested questions"):
    suggestions = [
        "When is stipend paid?",
        "How do I apply for leave?",
        "VPN access steps",
        "How to get a laptop?",
        "Where can I see public holidays?"
    ]
    st.write(", ".join(f"`{s}`" for s in suggestions))

user_q = st.chat_input("Ask a question…")
if user_q:
    st.session_state.chat.append(("user", user_q))

# Render chat
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

# Respond to the latest user turn
if st.session_state.chat and st.session_state.chat[-1][0] == "user":
    q = st.session_state.chat[-1][1]
    with st.chat_message("assistant"):
        q_aug = f"[{cat_filter}] {q}" if cat_filter != "All" else q
        q_emb = embedder.encode([q_aug]).tolist()
        res = collection.query(
            query_embeddings=q_emb,
            n_results=TOP_K,
            include=["distances","metadatas","documents"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits = list(zip(docs, metas, dists))

        final_ans, hits_used = answer_from_hits(hits)
        st.markdown(final_ans)

        if hits_used:
            with st.expander("Sources"):
                for i, (doc, meta, dist) in enumerate(hits_used, start=1):
                    sim = 1 - float(dist)
                    st.write(f"**{i}. {meta.get('question', '(FAQ)')}**")
                    st.write(doc)
                    st.caption(f"Category: {meta.get('category','')} • Similarity≈{sim:.2f}")

        # Feedback row
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("👍 Helpful", key=f"up_{len(st.session_state.chat)}"):
                log_feedback(q, final_ans, True, "")
                st.success("Thanks for the feedback!")
        with c2:
            if st.button("👎 Not helpful", key=f"down_{len(st.session_state.chat)}"):
                log_feedback(q, final_ans, False, "")
                st.warning("Logged. We’ll improve it.")
        with c3:
            fb = st.text_input("Optional note", key=f"fb_{len(st.session_state.chat)}", label_visibility="collapsed")
            if st.button("Save note", key=f"savenote_{len(st.session_state.chat)}"):
                log_feedback(q, final_ans, None, fb or "")
                st.info("Note saved.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips**")
st.sidebar.write("- Replace `faqs.csv` to rebuild the index automatically.")
st.sidebar.write("- Adjust `SIM_THRESHOLD`/`TOP_K` via env vars if matching feels off.")
st.sidebar.write("- Feedback is stored at `./logs/feedback.csv`.")
