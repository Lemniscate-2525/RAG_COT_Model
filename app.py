import html, os, re, json
from dataclasses import asdict, dataclass
from typing import List
import faiss, numpy as np, requests, streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

@dataclass
class Chunk:
    text: str
    title: str
    url: str
    post_id: int
    chunk_id: int

def clean(value):
    return re.sub(r"\s+", " ", html.unescape(BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True))).strip()

def split_chunks(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    result, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(size * .7), end)
            if boundary > start: end = boundary
        if text[start:end].strip(): result.append(text[start:end].strip())
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return result

@st.cache_resource
def embedder():
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

def fetch_chunks(site, pages=2, per_page=50):
    out, seen = [], set()
    for page in range(1, pages + 1):
        r = requests.get(site.rstrip("/") + "/wp-json/wp/v2/posts",
            params={"page":page,"per_page":min(100,per_page),"_fields":"id,link,title,content"}, timeout=20)
        if r.status_code == 400 and page > 1: break
        r.raise_for_status()
        batch = r.json()
        for post in batch:
            pid = int(post["id"])
            if pid in seen: continue
            seen.add(pid)
            title = clean(post.get("title",{}).get("rendered","")) or "Untitled"
            body = clean(post.get("content",{}).get("rendered",""))
            for i, text in enumerate(split_chunks(title + "\n\n" + body)):
                out.append(asdict(Chunk(text,title,post.get("link",site),pid,i)))
        if len(batch) < min(100,per_page): break
    if not out: raise ValueError("No posts found. Check the site URL and public WordPress REST API.")
    return out

def make_index(chunks):
    vec = np.asarray(embedder().encode([x["text"] for x in chunks], normalize_embeddings=True), dtype="float32")
    idx = faiss.IndexFlatIP(vec.shape[1])  # normalized vectors => cosine similarity
    idx.add(vec)
    return idx

def retrieve(question, chunks, index, k=3):
    if index is None or index.ntotal == 0: return []
    q = np.asarray(embedder().encode([question], normalize_embeddings=True), dtype="float32")
    scores, ids = index.search(q, min(max(1, int(k)), index.ntotal))
    result = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0 or i >= len(chunks): continue  # FAISS missing-neighbour sentinel
        item = dict(chunks[int(i)]); item["score"] = float(score); result.append(item)
    return result

def answer(question, docs):
    if not docs:
        return "No relevant source chunks were retrieved.", ["No evidence was retrieved; I cannot answer from this corpus."], []
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return ("LLM generation is not configured. Review the retrieved evidence below; this is not a generated answer.",
                ["Embedded the question.", f"Retrieved {len(docs)} chunks by cosine similarity.", "Inspect the cited passages; similarity is not proof of factual support."], docs)
    try:
        from openai import OpenAI
        sources = "\n\n".join(f"[S{i+1}] {d['title']} | {d['url']}\n{d['text']}" for i,d in enumerate(docs))
        prompt = f"""Answer only from these sources. If unsupported, say so. Return JSON keys: answer, evidence_steps (3-5 concise user-facing checks of evidence, not private chain-of-thought), used_sources (labels S1, S2...). Question: {question}\nSources:\n{sources}"""
        response = OpenAI(api_key=key).chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0.1,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":"Ground every claim in the provided sources. Give concise, verifiable evidence summaries, not hidden chain-of-thought."},{"role":"user","content":prompt}])
        data = json.loads(response.choices[0].message.content)
        used = set(data.get("used_sources", []))
        cited = [d for i,d in enumerate(docs) if not used or f"S{i+1}" in used]
        return str(data.get("answer","No answer returned.")), data.get("evidence_steps",[]), cited
    except Exception as e:
        return f"Generation failed ({e}). Retrieved passages are shown below instead.", ["No generated reasoning is claimed because the LLM call failed."], docs

st.set_page_config(page_title="WordPress RAG Evidence", page_icon="🔎", layout="wide")
st.title("WordPress RAG — Evidence & Reasoning")
st.caption("Fetch → clean → chunk with overlap → embed → retrieve → answer → inspect sources.")
with st.sidebar:
    site = st.text_input("WordPress site URL", os.getenv("WORDPRESS_SITE_URL","https://techcrunch.com"))
    pages = st.slider("Pages to fetch",1,10,2)
    per_page = st.slider("Posts per page",5,100,50,step=5)
    k = st.slider("Retrieved chunks",1,8,3)
    st.caption("Set OPENAI_API_KEY for generated answers and evidence-step summaries.")
    load = st.button("Fetch and index",type="primary")
if "chunks" not in st.session_state: st.session_state.chunks, st.session_state.index = [], None
if load:
    try:
        with st.spinner("Fetching, chunking and embedding posts..."):
            chunks = fetch_chunks(site,pages,per_page); index = make_index(chunks)
            st.session_state.chunks, st.session_state.index = chunks,index
        st.success(f"Indexed {len(chunks)} chunks.")
    except Exception as e: st.error(f"Indexing failed: {e}")
if st.session_state.index is not None:
    st.info(f"Index contains {len(st.session_state.chunks)} chunks.")
    q = st.text_input("Ask a question about these posts")
    if st.button("Ask",type="primary",disabled=not q.strip()):
        docs = retrieve(q.strip(),st.session_state.chunks,st.session_state.index,k)
        response, steps, cited = answer(q.strip(),docs)
        st.subheader("Answer"); st.write(response)
        st.subheader("Evidence trail")
        for i,step in enumerate(steps,1): st.markdown(f"{i}. {step}")
        st.subheader("Retrieved sources")
        if not cited: st.warning("No sources to display.")
        for i,d in enumerate(cited,1):
            with st.expander(f"{i}. {d['title']} · cosine={d.get('score',0):.3f}"):
                st.markdown(f"[Open source]({d['url']})"); st.write(d["text"])
else: st.info("Choose a WordPress URL and click Fetch and index.")
