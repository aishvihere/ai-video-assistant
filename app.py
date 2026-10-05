import html
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processing import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_questions, extract_decisions, extract_actionables
from core.rag_engine import ask_question, build_rag_pipeline

load_dotenv()

st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")

# ---------------- styling ----------------
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}

/* hero */
.hero {
    padding: 2.2rem 2rem; border-radius: 20px; margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 45%, #2563eb 100%);
    box-shadow: 0 10px 40px rgba(109, 40, 217, 0.25);
}
.hero h1 {margin: 0; font-size: 2.1rem; color: white; line-height: 1.2;}
.hero p {margin: .5rem 0 0; color: rgba(255,255,255,.8); font-size: 1.05rem;}

/* feature cards on landing */
.feature {
    background: #171b26; border: 1px solid #262c3d; border-radius: 16px;
    padding: 1.3rem; height: 100%;
}
.feature .icon {font-size: 1.8rem;}
.feature h4 {margin: .5rem 0 .3rem; color: #fff;}
.feature p {margin: 0; color: #9aa3b8; font-size: .92rem;}

/* stat cards */
.stat {
    background: #171b26; border: 1px solid #262c3d; border-radius: 14px;
    padding: 1rem 1.2rem; text-align: center;
}
.stat .num {font-size: 1.9rem; font-weight: 700; color: #a78bfa;}
.stat .label {color: #9aa3b8; font-size: .85rem;}

/* item cards */
.card {
    background: #171b26; border: 1px solid #262c3d; border-left: 4px solid #8b5cf6;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}
.card.green {border-left-color: #22c55e;}
.card.amber {border-left-color: #f59e0b;}
.card .title {font-weight: 600; font-size: 1.02rem; color: #fff; margin-bottom: .5rem;}
.card .quote {
    margin-top: .6rem; padding-left: .8rem; border-left: 2px solid #343b52;
    color: #9aa3b8; font-style: italic; font-size: .9rem;
}
.card .body {color: #c7cde0; font-size: .93rem; margin-top: .4rem;}

/* badges */
.badge {
    display: inline-block; padding: .15rem .65rem; border-radius: 999px;
    font-size: .78rem; margin-right: .4rem; background: #23283a; color: #c7cde0;
}
.badge.green {background: rgba(34,197,94,.15); color: #4ade80;}
.badge.amber {background: rgba(245,158,11,.15); color: #fbbf24;}
.badge.purple {background: rgba(139,92,246,.18); color: #c4b5fd;}

/* summary box */
.summary {
    background: #171b26; border: 1px solid #262c3d; border-radius: 14px;
    padding: 1.4rem 1.6rem; line-height: 1.7; color: #d6dbea;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: .4rem;}
.stTabs [data-baseweb="tab"] {
    background: #171b26; border-radius: 10px; padding: .5rem 1rem; border: 1px solid #262c3d;
}
.stTabs [aria-selected="true"] {background: #4c1d95 !important; border-color: #8b5cf6;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* buttons */
.stButton > button {border-radius: 10px; font-weight: 600;}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #7c3aed, #2563eb); border: none;
}

/* sidebar */
[data-testid="stSidebar"] {background: #12151e; border-right: 1px solid #1f2433;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- helpers ----------------
def esc(x) -> str:
    return html.escape(str(x))


def badge(text, kind=""):
    return f'<span class="badge {kind}">{esc(text)}</span>'


def run_pipeline_with_progress(source: str, language: str) -> dict:
    with st.status("Processing your video...", expanded=True) as status:
        st.write("⬇️ Downloading / splitting audio...")
        chunks = process_input(source)

        st.write("📝 Transcribing...")
        transcript = transcribe_all(chunks, language)

        st.write("🏷️ Writing title and summary...")
        title = generate_title(transcript)
        summary = summarize(transcript)

        st.write("✅ Extracting action items, decisions and questions...")
        action_items = extract_actionables(transcript)
        decisions = extract_decisions(transcript)
        questions = extract_questions(transcript)

        st.write("🔎 Building chat index...")
        rag_chain = build_rag_pipeline(transcript)

        status.update(label="Done!", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def stat_card(num, label):
    return f'<div class="stat"><div class="num">{num}</div><div class="label">{esc(label)}</div></div>'


def show_action_items(items):
    if not items:
        st.info("No action items found.")
        return
    for i, a in enumerate(items, 1):
        quote = ""
        if a.evidence and a.evidence != "Not specified":
            quote = f'<div class="quote">“{esc(a.evidence)}”</div>'
        st.markdown(
            f"""<div class="card">
                <div class="title">{i}. {esc(a.task)}</div>
                {badge("👤 " + a.owner, "purple")}{badge("📅 " + a.deadline)}
                {quote}
            </div>""",
            unsafe_allow_html=True,
        )


def show_decisions(items):
    if not items:
        st.info("No decisions found.")
        return
    for i, d in enumerate(items, 1):
        why = ""
        if d.rationale != "Not specified":
            why = f'<div class="body"><b>Why:</b> {esc(d.rationale)}</div>'
        st.markdown(
            f"""<div class="card green">
                <div class="title">{i}. {esc(d.decision)}</div>
                {badge("Decided by: " + d.decided_by, "green")}
                {why}
            </div>""",
            unsafe_allow_html=True,
        )


def show_questions(items):
    if not items:
        st.info("No questions found.")
        return
    for i, q in enumerate(items, 1):
        if q.answered:
            status_badge = badge("Answered", "green")
            ans = f'<div class="body"><b>Answer:</b> {esc(q.answer)}</div>'
            cls = "card green"
        else:
            status_badge = badge("Unanswered", "amber")
            ans, cls = "", "card amber"
        st.markdown(
            f"""<div class="{cls}">
                <div class="title">{i}. {esc(q.question)}</div>
                {status_badge}{badge("Asked by: " + q.asked_by)}
                {ans}
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------- state ----------------
if "result" not in st.session_state:
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## 🎬 AI Video Assistant")
    st.caption("Summaries, action items, decisions, and chat for any video or meeting.")
    st.divider()

    input_mode = st.radio("Input type", ["YouTube URL", "Upload file"], horizontal=True)
    language = st.selectbox("Language", ["english", "hinglish"])

    source = None
    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a"],
        )
        if uploaded is not None:
            source = save_upload_to_temp(uploaded)

    if st.button("✨ Process", type="primary", use_container_width=True, disabled=not source):
        st.session_state.messages = []
        try:
            st.session_state.result = run_pipeline_with_progress(source, language)
        except Exception as e:
            st.session_state.result = None
            st.error(f"Something went wrong: {e}")

    if st.session_state.result and st.button("Clear", use_container_width=True):
        st.session_state.result = None
        st.session_state.messages = []
        st.rerun()


# ---------------- main area ----------------
result = st.session_state.result

if result is None:
    st.markdown(
        """<div class="hero">
            <h1>Turn any video into notes you can chat with</h1>
            <p>Paste a YouTube link or upload a recording in the sidebar, then hit Process.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    features = [
        ("📝", "Smart summary", "A clean title and summary of everything that was said."),
        ("✅", "Actions & decisions", "Who needs to do what, what was decided, and what's still open."),
        ("💬", "Chat with it", "Ask questions and get answers grounded in the transcript."),
    ]
    for col, (icon, title, text) in zip((c1, c2, c3), features):
        col.markdown(
            f'<div class="feature"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f"""<div class="hero"><h1>{esc(result['title'])}</h1>
        <p>Processed and ready. Explore the tabs below or chat with the video.</p></div>""",
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(stat_card(len(result["action_items"]), "Action items"), unsafe_allow_html=True)
    s2.markdown(stat_card(len(result["key_decisions"]), "Decisions"), unsafe_allow_html=True)
    s3.markdown(stat_card(len(result["open_questions"]), "Questions"), unsafe_allow_html=True)
    s4.markdown(stat_card(f"{len(result['transcript'].split()):,}", "Words"), unsafe_allow_html=True)
    st.write("")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📝 Summary", "✅ Actions", "🔑 Decisions", "❓ Questions", "📄 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f'<div class="summary">{esc(result["summary"]).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)

    with tab_actions:
        show_action_items(result["action_items"])

    with tab_decisions:
        show_decisions(result["key_decisions"])

    with tab_questions:
        show_questions(result["open_questions"])

    with tab_transcript:
        st.download_button("⬇️ Download transcript", result["transcript"], file_name="transcript.txt")
        st.text_area("Full transcript", result["transcript"], height=400, label_visibility="collapsed")

    with tab_chat:
        suggestions = ["Summarize the key points", "What were the main takeaways?", "Who said what about the next steps?"]
        pending = None
        if not st.session_state.messages:
            st.caption("Try one of these, or type your own:")
            cols = st.columns(len(suggestions))
            for col, s in zip(cols, suggestions):
                if col.button(s, use_container_width=True):
                    pending = s

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask a question about the video...") or pending
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(result["rag_chain"], question)
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})