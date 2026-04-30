"""
server.py — Railway deployment (LangGraph Multi-Agent Edition)
==============================================================
Architecture: LangGraph multi-agent pipeline (NO intent/orchestration agent)

  ┌──────────────┐
  │ router_agent │  ← uses get_current_time tool, routes by rule
  └──────┬───────┘
         │  conditional edge
   ┌─────┴──────┬──────────────┬─────────────┐
   ▼            ▼              ▼             ▼
news_agent  syllabus_agent  rag_agent  greeting_agent
   │            │              │             │
   └────────────┴──────────────┴─────────────┘
                        │
                  reply_agent  ← Groq LLM formats TTS-friendly reply
                        │
                       END

Tools:
  • get_current_time  → IST hour, checks if 8 AM
  • fetch_news        → scrapes aiml.pccoepune.com home for NEWS & ANNOUNCEMENTS
  • fetch_syllabus    → returns info for hardcoded SY/TY/BE PDFs

Env vars (Railway dashboard):
    GROQ_API_KEY, CRAWL_SECRET
"""

import os
import re
import json
import pickle
import hashlib
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Annotated, Literal
from pathlib import Path
from collections import deque

import httpx
import numpy as np
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
ENCODINGS_FILE  = "encodings.pkl"
WEBSITE_CACHE   = "website_chunks.json"
FACE_THRESHOLD  = 0.5
CHUNK_SIZE      = 400
MAX_RAG_CHUNKS  = 5
CRAWL_MAX_PAGES = 60
GROQ_MODEL      = "llama-3.1-8b-instant"
IST             = timezone(timedelta(hours=5, minutes=30))

# Hardcoded syllabus PDFs — keyword → (label, url)
SYLLABUS_MAP = {
    "sy":          ("Second Year (SY) B.Tech AI&ML 2022",         "https://aiml.pccoepune.com/assets/docs/Syllabus/SY_BTech_AIML_Syllabus_2022.pdf"),
    "second year": ("Second Year (SY) B.Tech AI&ML 2022",         "https://aiml.pccoepune.com/assets/docs/Syllabus/SY_BTech_AIML_Syllabus_2022.pdf"),
    "ty":          ("Third Year (TY) B.Tech CSE(AI-ML)",          "https://aiml.pccoepune.com/assets/docs/Syllabus/TY-B.Tech-CSE(AI-ML).pdf"),
    "third year":  ("Third Year (TY) B.Tech CSE(AI-ML)",          "https://aiml.pccoepune.com/assets/docs/Syllabus/TY-B.Tech-CSE(AI-ML).pdf"),
    "be":          ("Final Year (BE) UG CSE AI&ML BTech 2024-25", "https://aiml.pccoepune.com/assets/docs/Syllabus/UG-CSE(AI%26ML)-BTech-24-25.pdf"),
    "final year":  ("Final Year (BE) UG CSE AI&ML BTech 2024-25", "https://aiml.pccoepune.com/assets/docs/Syllabus/UG-CSE(AI%26ML)-BTech-24-25.pdf"),
    "fourth year": ("Final Year (BE) UG CSE AI&ML BTech 2024-25", "https://aiml.pccoepune.com/assets/docs/Syllabus/UG-CSE(AI%26ML)-BTech-24-25.pdf"),
}

SEED_URLS = [
    "https://aiml.pccoepune.com/",
    "https://aiml.pccoepune.com/aiml-hod",
    "https://aiml.pccoepune.com/faculty-profile",
    "https://aiml.pccoepune.com/about",
    "https://aiml.pccoepune.com/research-area-sigs",
    "https://aiml.pccoepune.com/stud-activities",
    "https://aiml.pccoepune.com/placement",
    "https://aiml.pccoepune.com/contact",
]

GREETING_MAP = {
    "good morning"  : "Good morning, {name}! Have a great day ahead!",
    "good afternoon": "Good afternoon, {name}! Hope your day is going well!",
    "good evening"  : "Good evening, {name}! Hope you had a wonderful day!",
    "hello"         : "Hello, {name}! Nice to meet you!",
    "hi"            : "Hi there, {name}! Great to see you!",
    "how are you"   : "I am doing great, {name}! Hope you are too!",
    "bye"           : "Goodbye, {name}! See you soon!",
    "good night"    : "Good night, {name}! Sweet dreams!",
    "thank you"     : "You are welcome, {name}! Happy to help!",
    "thanks"        : "Anytime, {name}! Take care!",
}

# ─── Globals ──────────────────────────────────────────────────────────────────
groq_client: Optional[Groq]     = None
face_encodings: list            = []
face_names: list                = []
rag_chunks: list[str]           = []
rag_vectors: Optional[np.ndarray] = None

# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS  (standalone functions — called explicitly inside agents)
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_current_time() -> dict:
    """Returns IST hour and whether it is 8 AM."""
    now = datetime.now(IST)
    return {
        "hour":     now.hour,
        "time_str": now.strftime("%I:%M %p IST"),
        "is_8am":   now.hour == 8,
    }


async def tool_fetch_news() -> str:
    """Scrapes NEWS & ANNOUNCEMENTS from aiml.pccoepune.com home page."""
    url = "https://aiml.pccoepune.com/"
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        news_items = []

        # Strategy 1: headings containing news/announcement
        for heading in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
            if re.search(r"news|announcement", heading.get_text(), re.I):
                sib = heading.find_next_sibling()
                if sib:
                    for el in sib.find_all(["li","p","a"]):
                        t = el.get_text(strip=True)
                        if t and len(t) > 10:
                            news_items.append(t)

        # Strategy 2: class-name hints
        if not news_items:
            for tag in soup.find_all(class_=re.compile(r"news|announce|marquee|notice", re.I)):
                for el in tag.find_all(["li","p","a","span"]):
                    t = el.get_text(strip=True)
                    if t and len(t) > 15:
                        news_items.append(t)

        # Strategy 3: marquee tags
        if not news_items:
            for m in soup.find_all("marquee"):
                for part in m.get_text(separator="|", strip=True).split("|"):
                    if part.strip():
                        news_items.append(part.strip())

        # Deduplicate
        seen, unique = set(), []
        for item in news_items:
            if item not in seen:
                seen.add(item)
                unique.append(item)

        if unique:
            lines = "\n".join(f"* {i}" for i in unique[:15])
            return f"Latest News and Announcements from AIML Department:\n{lines}"
        return "The AIML department website was reached but no news items were found. Please visit aiml.pccoepune.com directly."
    except Exception as e:
        log.error("tool_fetch_news error: %s", e)
        return "Sorry, I could not fetch the news right now due to a network error."


async def tool_fetch_syllabus(user_text: str) -> str:
    """Matches year keyword and returns syllabus label + URL."""
    kw = user_text.lower()
    for key, (label, url) in SYLLABUS_MAP.items():
        if key in kw:
            return f"The syllabus for {label} is available at: {url}"

    # No match — list all options
    seen, lines = set(), []
    for label, _ in SYLLABUS_MAP.values():
        if label not in seen:
            seen.add(label)
            lines.append(f"* {label}")
    return (
        "Please specify which year syllabus you need. Available options are:\n"
        + "\n".join(lines)
        + "\nFor example, say: show me SY syllabus, or TY syllabus, or BE syllabus."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE & AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    name:        str
    user_text:   str
    messages:    Annotated[list[BaseMessage], add_messages]
    route:       str          # "news" | "syllabus" | "rag" | "greeting"
    tool_result: str          # raw output from the domain tool
    final_reply: str          # TTS-ready formatted reply


# ── Agent 1: Router ───────────────────────────────────────────────────────────
async def router_agent(state: AgentState) -> AgentState:
    """
    Rule-based router. Calls get_current_time tool.
    8 AM  → news
    syllabus keywords → syllabus
    greeting keywords → greeting
    everything else   → rag
    """
    log.info("[RouterAgent] text=%r", state["user_text"])
    time_info = tool_get_current_time()   # ← TOOL CALL
    log.info("[RouterAgent] get_current_time → %s  is_8am=%s", time_info["time_str"], time_info["is_8am"])

    text = state["user_text"].lower()

    if time_info["is_8am"]:
        route = "news"
    elif any(kw in text for kw in ["syllabus", "curriculum", "subjects", "course"]):
        route = "syllabus"
    elif any(kw in text for kw in GREETING_MAP):
        route = "greeting"
    else:
        route = "rag"

    log.info("[RouterAgent] route=%s", route)
    return {
        **state,
        "route": route,
        "messages": [HumanMessage(content=f"[RouterAgent] route={route} time={time_info['time_str']}")],
    }


# ── Agent 2a: News ────────────────────────────────────────────────────────────
async def news_agent(state: AgentState) -> AgentState:
    """Calls fetch_news tool and stores raw result."""
    log.info("[NewsAgent] fetching...")
    result = await tool_fetch_news()      # ← TOOL CALL
    log.info("[NewsAgent] got %d chars", len(result))
    return {
        **state,
        "tool_result": result,
        "messages": [AIMessage(content=f"[NewsAgent] fetched {len(result)} chars")],
    }


# ── Agent 2b: Syllabus ────────────────────────────────────────────────────────
async def syllabus_agent(state: AgentState) -> AgentState:
    """Calls fetch_syllabus tool."""
    log.info("[SyllabusAgent] text=%r", state["user_text"])
    result = await tool_fetch_syllabus(state["user_text"])   # ← TOOL CALL
    log.info("[SyllabusAgent] done")
    return {
        **state,
        "tool_result": result,
        "messages": [AIMessage(content="[SyllabusAgent] syllabus info ready")],
    }


# ── Agent 2c: RAG ─────────────────────────────────────────────────────────────
def rag_agent(state: AgentState) -> AgentState:
    """Vector similarity retrieval from crawled website chunks."""
    log.info("[RAGAgent] query=%r", state["user_text"])
    result = retrieve_chunks(state["user_text"])
    if not result:
        result = "No relevant information found in the knowledge base."
    log.info("[RAGAgent] retrieved %d chars", len(result))
    return {
        **state,
        "tool_result": result,
        "messages": [AIMessage(content=f"[RAGAgent] retrieved {len(result)} chars")],
    }


# ── Agent 2d: Greeting ────────────────────────────────────────────────────────
def greeting_agent(state: AgentState) -> AgentState:
    """No tool needed — pure keyword lookup."""
    log.info("[GreetingAgent] handling greeting")
    text_lower = state["user_text"].lower()
    reply = f"Hello, {state['name']}! How can I help you today?"
    for trigger, template in GREETING_MAP.items():
        if trigger in text_lower:
            reply = template.format(name=state["name"])
            break
    return {
        **state,
        "tool_result": reply,
        "final_reply": reply,   # skip reply_agent for greetings
        "messages": [AIMessage(content="[GreetingAgent] greeting prepared")],
    }


# ── Agent 3: Reply ────────────────────────────────────────────────────────────
async def reply_agent(state: AgentState) -> AgentState:
    """
    Uses Groq to turn raw tool_result into a concise TTS-friendly reply.
    Falls back to tool_result directly if LLM unavailable.
    """
    # Greetings already have final_reply set
    if state.get("final_reply"):
        return state

    log.info("[ReplyAgent] formatting reply...")
    raw    = state.get("tool_result", "")
    name   = state["name"]

    if not groq_client or not raw:
        reply = f"{name}, {raw}" if name != "Guest" and name not in raw else raw
        return {**state, "final_reply": reply}

    system = (
        "You are a voice assistant for PCCOE AI and ML department. "
        "Your reply will be spoken aloud. Keep it to 2-4 sentences. "
        "No markdown, no bullet points, no URLs. Be natural and friendly. "
        f"Always address the person as {name}."
    )
    user = (
        f'The user said: "{state["user_text"]}"\n\n'
        f"Relevant information:\n{raw}\n\n"
        "Give a concise spoken reply."
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            max_tokens=200,
            temperature=0.4,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        log.error("[ReplyAgent] Groq error: %s", e)
        reply = raw[:500]

    if name != "Guest" and name not in reply:
        reply = f"{name}, {reply}"

    log.info("[ReplyAgent] reply ready (%d chars)", len(reply))
    return {
        **state,
        "final_reply": reply,
        "messages": [AIMessage(content="[ReplyAgent] reply formatted")],
    }


# ── Conditional routing function ──────────────────────────────────────────────
def pick_domain_agent(state: AgentState) -> Literal["news_agent","syllabus_agent","rag_agent","greeting_agent"]:
    return {
        "news":     "news_agent",
        "syllabus": "syllabus_agent",
        "rag":      "rag_agent",
        "greeting": "greeting_agent",
    }.get(state["route"], "rag_agent")


# ── Build & compile graph ─────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router_agent",   router_agent)
    g.add_node("news_agent",     news_agent)
    g.add_node("syllabus_agent", syllabus_agent)
    g.add_node("rag_agent",      rag_agent)
    g.add_node("greeting_agent", greeting_agent)
    g.add_node("reply_agent",    reply_agent)

    g.set_entry_point("router_agent")
    g.add_conditional_edges("router_agent", pick_domain_agent)

    for node in ["news_agent", "syllabus_agent", "rag_agent", "greeting_agent"]:
        g.add_edge(node, "reply_agent")

    g.add_edge("reply_agent", END)
    return g.compile()


agent_graph = build_graph()
log.info("LangGraph compiled: 6 agents, 3 tools ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="RPi AI Server — LangGraph Multi-Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    global groq_client, face_encodings, face_names

    key = os.getenv("GROQ_API_KEY", "")
    if key:
        groq_client = Groq(api_key=key)
        log.info("Groq ready")
    else:
        log.warning("GROQ_API_KEY missing — LLM disabled")

    if Path(ENCODINGS_FILE).exists():
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        face_encodings = data["encodings"]
        face_names     = data["names"]
        log.info("Face encodings loaded: %s", face_names)
    else:
        log.warning("%s not found — face ID disabled", ENCODINGS_FILE)

    if Path(WEBSITE_CACHE).exists():
        load_rag_from_cache()
    else:
        asyncio.create_task(crawl_and_index())


# ─── Models ───────────────────────────────────────────────────────────────────
class FaceRequest(BaseModel):
    encoding: list[float]

class ChatRequest(BaseModel):
    name: str
    text: str

class FaceResponse(BaseModel):
    name: str
    confidence: float

class ChatResponse(BaseModel):
    reply: str
    source: str


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/identify", response_model=FaceResponse)
def identify_face(req: FaceRequest):
    if not face_encodings:
        return FaceResponse(name="Guest", confidence=0.0)
    query     = np.array(req.encoding, dtype=np.float64)
    distances = [np.linalg.norm(query - np.array(e, dtype=np.float64)) for e in face_encodings]
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    if best_dist > FACE_THRESHOLD:
        return FaceResponse(name="Guest", confidence=round(1 - best_dist, 3))
    return FaceResponse(name=face_names[best_idx].replace("_"," ").title(),
                        confidence=round(1 - best_dist, 3))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    name = req.name or "Guest"
    text = req.text.strip()
    if not text:
        return ChatResponse(reply=f"Sorry {name}, I didn't catch that.", source="fallback")

    log.info(">>> /chat  name=%s  text=%s", name, text)

    initial: AgentState = {
        "name":        name,
        "user_text":   text,
        "messages":    [HumanMessage(content=text)],
        "route":       "",
        "tool_result": "",
        "final_reply": "",
    }

    try:
        result = await agent_graph.ainvoke(initial)
        reply  = result.get("final_reply") or f"Sorry {name}, I could not process that."
        route  = result.get("route", "rag")
        log.info("<<< route=%s reply=%s", route, reply[:80])
        return ChatResponse(reply=reply, source=route)
    except Exception as e:
        log.error("Graph error: %s", e)
        return ChatResponse(reply=f"Sorry {name}, something went wrong.", source="error")


@app.post("/crawl")
async def trigger_crawl(background_tasks: BackgroundTasks, secret: str = ""):
    if secret != os.getenv("CRAWL_SECRET", "changeme"):
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(crawl_and_index)
    return {"status": "crawl started"}


@app.get("/health")
def health():
    t = tool_get_current_time()
    return {
        "faces":      len(face_names),
        "rag_chunks": len(rag_chunks),
        "groq":       groq_client is not None,
        "graph":      "langgraph_multi_agent",
        "agents":     ["router_agent","news_agent","syllabus_agent","rag_agent","greeting_agent","reply_agent"],
        "tools":      ["get_current_time","fetch_news","fetch_syllabus"],
        "ist_time":   t["time_str"],
        "is_8am":     t["is_8am"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RAG UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def build_local_index(chunks: list[str]):
    global rag_vectors
    if chunks:
        rag_vectors = np.stack([text_to_vector(c) for c in chunks])
        log.info("Local vector index: %d chunks", len(chunks))

def retrieve_chunks(query: str, k: int = MAX_RAG_CHUNKS) -> str:
    if rag_vectors is None or not rag_chunks:
        return ""
    scores = rag_vectors @ text_to_vector(query)
    idxs   = np.argsort(scores)[::-1][:min(k, len(rag_chunks))]
    return "\n\n".join(rag_chunks[i] for i in idxs)

def load_rag_from_cache():
    global rag_chunks
    with open(WEBSITE_CACHE, "r", encoding="utf-8") as f:
        rag_chunks = json.load(f)
    build_local_index(rag_chunks)
    log.info("RAG loaded: %d chunks", len(rag_chunks))

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < size:
            cur += " " + s
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = s
    if cur.strip(): chunks.append(cur.strip())
    return chunks

async def crawl_and_index():
    global rag_chunks
    log.info("Crawling...")
    all_chunks, visited, q = [], set(), deque(SEED_URLS)
    base = "aiml.pccoepune.com"
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        while q and len(visited) < CRAWL_MAX_PAGES:
            url = q.popleft()
            if url in visited: continue
            visited.add(url)
            try:
                resp = await client.get(url)
                if resp.status_code != 200: continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for t in soup(["script","style","nav","footer","header","aside"]): t.decompose()
                text = re.sub(r'\s{2,}', ' ', soup.get_text(separator=" ", strip=True)).strip()
                if len(text) > 100:
                    chunks = chunk_text(text)
                    all_chunks.extend(f"[Source: {url}]\n{c}" for c in chunks)
                    log.info("  ✓ %s → %d chunks", url, len(chunks))
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    full = f"https://{base}{href}" if href.startswith("/") else href
                    if base in full:
                        full = full.split("#")[0].rstrip("/")
                        if full not in visited: q.append(full)
                await asyncio.sleep(0.3)
            except Exception as e:
                log.warning("  ✗ %s: %s", url, e)
    rag_chunks = all_chunks
    with open(WEBSITE_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    build_local_index(rag_chunks)
    log.info("Crawl done: %d pages, %d chunks", len(visited), len(rag_chunks))
