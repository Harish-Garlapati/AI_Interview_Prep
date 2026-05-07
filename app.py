import html
import json
import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components

try:
    from rag_pipeline import (
        load_pdf,
        split_documents,
        create_vectorstore,
        create_conversation_chain,
        GROQ_DEFAULT_MODEL,
        GEMINI_DEFAULT_MODEL,
        OLLAMA_DEFAULT_MODEL,
        analyze_jd_match,
        generate_prep_report,
        get_profile_insights,
        score_answer,
    )
except ImportError as exc:
    st.error(
        "A required package is missing. Install dependencies with "
        "`pip install -r requirements.txt`, then restart Streamlit."
    )
    st.exception(exc)
    st.stop()


st.set_page_config(
    page_title="AI Interview Prep Assistant",
    page_icon="🎯",
    layout="wide",
)


def add_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f8fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #637083;
            --accent: #1f7a8c;
            --accent-2: #f6b73c;
            --line: #e2e8f0;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31, 122, 140, 0.12), transparent 32rem),
                linear-gradient(180deg, #fbfcfe 0%, var(--bg) 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #101828;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero {
            padding: 1.4rem 0 0.8rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }

        .hero h1 {
            font-size: 2.3rem;
            line-height: 1.15;
            margin: 0 0 0.4rem;
            letter-spacing: 0;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.03rem;
            max-width: 62rem;
            margin: 0;
        }

        .status-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            min-height: 7.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }

        .status-card h3 {
            font-size: 0.9rem;
            color: var(--muted);
            margin: 0 0 0.4rem;
            font-weight: 600;
        }

        .status-card p {
            font-size: 1.35rem;
            margin: 0;
            font-weight: 700;
            color: var(--ink);
        }

        .tip-box {
            border-left: 4px solid var(--accent);
            background: #eef9fb;
            padding: 0.85rem 1rem;
            border-radius: 6px;
            color: #164e5b;
            margin: 0.5rem 0 1rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.82);
        }

        div[data-testid="stChatMessage"],
        div[data-testid="stChatMessage"] *,
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stMarkdownContainer"] * {
            color: var(--ink) !important;
        }

        div[data-testid="stChatMessage"] code,
        div[data-testid="stMarkdownContainer"] code {
            color: #0f766e !important;
            background: #ecfeff !important;
        }

        div[data-testid="stChatMessage"] pre,
        div[data-testid="stMarkdownContainer"] pre {
            background: #111827 !important;
            border-radius: 8px;
        }

        div[data-testid="stChatMessage"] pre *,
        div[data-testid="stMarkdownContainer"] pre * {
            color: #f8fafc !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {
            border-radius: 6px;
            border: 1px solid #176979;
            background: var(--accent);
            color: white;
            font-weight: 650;
        }

        .stButton > button:hover,
        button[kind="primary"]:hover {
            border-color: #145967;
            background: #176979;
            color: white;
        }

        textarea, input {
            border-radius: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "conversation_chain": None,
        "chat_history": [],
        "doc_stats": None,
        "pending_question": "",
        "mock_question": "Tell me about yourself.",
        "mock_feedback": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_documents(resume_file, jd_file, llm_provider, llm_model, api_key):
    temp_paths = []

    try:
        with st.spinner("Reading PDFs and building free local RAG index..."):
            all_documents = []

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_resume:
                temp_resume.write(resume_file.read())
                resume_path = temp_resume.name
                temp_paths.append(resume_path)

            all_documents.extend(load_pdf(resume_path, source_label="Resume"))

            if jd_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_jd:
                    temp_jd.write(jd_file.read())
                    jd_path = temp_jd.name
                    temp_paths.append(jd_path)

                all_documents.extend(load_pdf(jd_path, source_label="Job Description"))

            if not all_documents:
                st.warning("No readable text was found in the uploaded PDF.")
                return

            chunks = split_documents(all_documents)
            if not chunks:
                st.warning("No usable text chunks were created from the PDF.")
                return

            vectorstore = create_vectorstore(chunks)
            st.session_state.conversation_chain = create_conversation_chain(
                vectorstore=vectorstore,
                llm_provider=llm_provider,
                llm_model=llm_model,
                api_key=api_key,
            )
            st.session_state.chat_history = []
            st.session_state.doc_stats = {
                "pages": len(all_documents),
                "chunks": len(chunks),
                "mode": f"RAG + {provider_label(llm_provider)}",
            }

            st.success("Documents processed successfully.")
    except Exception as exc:
        st.error("Document processing failed. Check that the uploaded PDFs contain selectable text.")
        st.exception(exc)
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def ask_assistant(question):
    if st.session_state.conversation_chain is None:
        st.warning("Please upload and process your resume first.")
        return

    clean_question = question.strip()
    if not clean_question:
        return

    with st.spinner("Preparing a focused answer..."):
        response = st.session_state.conversation_chain.invoke({"question": clean_question})
        answer = response.get("answer", "I could not generate an answer.")

    st.session_state.chat_history.append(
        {
            "question": clean_question,
            "answer": answer,
        }
    )


def get_vectorstore():
    chain = st.session_state.conversation_chain
    if chain is None:
        return None
    return getattr(chain, "vectorstore", None)


def set_question(question):
    st.session_state.pending_question = question


def render_metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)


def provider_label(provider):
    labels = {
        "groq": "Groq LLM",
        "gemini": "Gemini LLM",
        "ollama": "Ollama Local LLM",
        "offline": "Offline Coach",
        "auto": "Auto LLM",
    }
    return labels.get(provider, "Auto LLM")


def get_config_value(name, default=""):
    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        value = ""

    return value or os.getenv(name, default)


def voice_tools(last_answer):
    safe_answer = json.dumps(last_answer or "")
    components.html(
        f"""
        <div style="font-family: Inter, Arial, sans-serif; border:1px solid #e2e8f0; border-radius:8px; padding:14px; background:#ffffff;">
          <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
            <button id="listen" style="border:0; border-radius:6px; padding:9px 12px; background:#1f7a8c; color:white; font-weight:700;">Start voice input</button>
            <button id="stop" style="border:1px solid #cbd5e1; border-radius:6px; padding:9px 12px; background:#f8fafc; color:#172033;">Stop</button>
            <button id="speak" style="border:0; border-radius:6px; padding:9px 12px; background:#f6b73c; color:#172033; font-weight:700;">Read last answer</button>
          </div>
          <textarea id="transcript" placeholder="Voice transcript appears here. Copy it into the question box above." style="width:100%; min-height:86px; border:1px solid #cbd5e1; border-radius:6px; padding:10px; color:#172033;"></textarea>
          <div id="status" style="font-size:12px; color:#637083; margin-top:8px;">Uses your browser's free speech recognition and text-to-speech when supported.</div>
        </div>
        <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const status = document.getElementById("status");
        const transcript = document.getElementById("transcript");
        let recognition = null;

        if (SpeechRecognition) {{
          recognition = new SpeechRecognition();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = "en-IN";

          recognition.onresult = (event) => {{
            let text = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {{
              text += event.results[i][0].transcript;
            }}
            transcript.value = text.trim();
          }};
          recognition.onerror = (event) => status.textContent = "Voice input error: " + event.error;
        }} else {{
          status.textContent = "Voice input is not supported in this browser. Text-to-speech may still work.";
        }}

        document.getElementById("listen").onclick = () => {{
          if (recognition) {{
            recognition.start();
            status.textContent = "Listening...";
          }}
        }};

        document.getElementById("stop").onclick = () => {{
          if (recognition) {{
            recognition.stop();
            status.textContent = "Stopped. Copy the transcript into the question box.";
          }}
        }};

        document.getElementById("speak").onclick = () => {{
          const answer = {safe_answer};
          if (!answer) {{
            status.textContent = "Ask a question first, then I can read the answer.";
            return;
          }}
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(answer.replace(/[#*_`]/g, ""));
          utterance.lang = "en-IN";
          utterance.rate = 0.95;
          window.speechSynthesis.speak(utterance);
        }};
        </script>
        """,
        height=220,
    )


add_styles()
init_state()

with st.sidebar:
    st.header("Upload Documents")
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    jd_file = st.file_uploader("Upload Job Description PDF (optional)", type=["pdf"])

    st.divider()
    st.header("LLM Settings")
    provider_choice = st.selectbox(
        "Answer engine",
        [
            "Groq free tier - fastest",
            "Gemini free tier",
            "Ollama local",
            "Auto",
            "Offline fallback only",
        ],
    )

    provider_map = {
        "Groq free tier - fastest": "groq",
        "Gemini free tier": "gemini",
        "Ollama local": "ollama",
        "Auto": "auto",
        "Offline fallback only": "offline",
    }
    llm_provider = provider_map[provider_choice]

    default_model = {
        "groq": get_config_value("GROQ_MODEL", GROQ_DEFAULT_MODEL),
        "gemini": get_config_value("GEMINI_MODEL", GEMINI_DEFAULT_MODEL),
        "ollama": get_config_value("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL),
        "auto": "",
        "offline": "",
    }[llm_provider]

    llm_model = st.text_input(
        "Model",
        value=default_model,
        disabled=llm_provider in ["auto", "offline"],
    )

    api_key = ""
    if llm_provider in ["groq", "gemini"]:
        key_name = "GROQ_API_KEY" if llm_provider == "groq" else "GEMINI_API_KEY"
        saved_key = get_config_value(key_name)

        if saved_key:
            api_key = saved_key
            st.success(f"{provider_label(llm_provider)} key loaded from app secrets.")
        else:
            key_label = "Groq API key" if llm_provider == "groq" else "Gemini API key"
            api_key = st.text_input(key_label, type="password")
    elif llm_provider == "auto":
        api_key = get_config_value("GROQ_API_KEY") or get_config_value("GEMINI_API_KEY")
        auto_has_key = bool(api_key)
        if auto_has_key:
            st.success("Auto mode found an LLM key in app secrets.")
        else:
            st.caption("Auto checks GROQ_API_KEY, GEMINI_API_KEY, then local Ollama.")
    elif llm_provider == "ollama":
        st.caption("Start Ollama locally before asking: `ollama run mistral`.")

    st.divider()
    process_button = st.button("Process Documents", use_container_width=True)
    clear_button = st.button("Clear Chat", use_container_width=True)

    st.divider()
    st.caption("Free stack: PyPDF + TF-IDF RAG + free-tier/local LLM + browser voice tools.")

if clear_button:
    st.session_state.chat_history = []
    st.session_state.conversation_chain = None
    st.session_state.doc_stats = None
    st.session_state.mock_feedback = None
    st.success("Chat cleared.")

if process_button:
    if resume_file is None:
        st.warning("Please upload a resume PDF.")
    elif llm_provider in ["groq", "gemini"] and not api_key:
        st.warning("Please enter the selected LLM API key, or choose Ollama/Offline fallback.")
    else:
        process_documents(resume_file, jd_file, llm_provider, llm_model, api_key)


st.markdown(
    """
    <div class="hero">
      <h1>AI Interview Preparation Assistant</h1>
      <p>Upload your resume, then get direct GenAI interview prep, aptitude practice, project explanations, and resume feedback using free local RAG.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = st.session_state.doc_stats or {"pages": 0, "chunks": 0, "mode": "Not processed"}
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="status-card"><h3>Document Pages</h3><p>{stats["pages"]}</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="status-card"><h3>RAG Chunks</h3><p>{stats["chunks"]}</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="status-card"><h3>Mode</h3><p>{html.escape(str(stats["mode"]))}</p></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="tip-box">Try: "Prepare me for a GenAI Developer interview", "Review my resume", "Give aptitude practice", or "Ask me mock interview questions".</div>',
    unsafe_allow_html=True,
)

vectorstore = get_vectorstore()
dashboard_tab, jd_tab, mock_tab, report_tab = st.tabs(
    ["Prep Dashboard", "JD Match", "Mock Interview", "Report"]
)

with dashboard_tab:
    if vectorstore is None:
        st.info("Process your resume to unlock profile insights.")
    else:
        insights = get_profile_insights(vectorstore)
        st.markdown("### Candidate Snapshot")
        st.write(f"**Core skills:** {insights['skills']}")
        st.write(f"**Education:** {insights['education']}")
        with st.expander("Project evidence"):
            st.write(insights["projects"])
        with st.expander("Experience evidence"):
            st.write(insights["experience"])

        st.markdown("### Fast Prep Actions")
        prep_cols = st.columns(4)
        prep_actions = [
            ("Self intro", "Create a strong 60-second self introduction from my resume."),
            ("RAG basics", "Explain RAG, embeddings, chunking, and vector search for my interview."),
            ("Project story", "Prepare an interview answer for my strongest GenAI project."),
            ("HR answers", "Prepare HR interview answers based on my profile."),
        ]
        for col, (label, prompt) in zip(prep_cols, prep_actions):
            with col:
                if st.button(label, use_container_width=True):
                    set_question(prompt)

with jd_tab:
    if vectorstore is None:
        st.info("Process your resume first. Upload a JD PDF too for matching.")
    else:
        jd_match = analyze_jd_match(vectorstore)
        if not jd_match["available"]:
            st.info(jd_match["summary"])
        else:
            st.metric("JD Match Score", f"{jd_match['score']}%")
            st.write(jd_match["summary"])
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Matched Keywords")
                st.write(", ".join(jd_match["matched"]) or "None detected")
            with col_b:
                st.markdown("#### Missing Keywords")
                st.write(", ".join(jd_match["missing"]) or "None detected")
            if st.button("Generate JD-focused interview prep", use_container_width=True):
                set_question("Based on my resume and job description, create a JD-focused interview preparation plan.")

with mock_tab:
    if vectorstore is None:
        st.info("Process your resume to start mock interview practice.")
    else:
        question_bank = [
            "Tell me about yourself.",
            "Explain your AI Chatbot with RAG project.",
            "What is RAG and why is it useful?",
            "How do you reduce hallucinations in an LLM app?",
            "Explain your internship work as an Associate Data Scientist.",
            "What is the difference between SQL WHERE and HAVING?",
            "Solve this aptitude question: A value increases by 20% and then decreases by 20%. What is the net change?",
        ]
        st.session_state.mock_question = st.selectbox(
            "Mock question",
            question_bank,
            index=question_bank.index(st.session_state.mock_question)
            if st.session_state.mock_question in question_bank
            else 0,
        )
        mock_answer = st.text_area(
            "Type your spoken/practiced answer here",
            height=140,
            placeholder="Answer as you would in an interview. Then click Evaluate Answer.",
        )
        if st.button("Evaluate Answer", type="primary"):
            if mock_answer.strip():
                st.session_state.mock_feedback = score_answer(
                    st.session_state.mock_question,
                    mock_answer,
                    vectorstore,
                )
            else:
                st.warning("Write your answer first.")

        feedback = st.session_state.mock_feedback
        if feedback:
            render_metric_row(
                [
                    ("Overall", f"{feedback['overall']}%"),
                    ("Relevance", f"{feedback['relevance']}%"),
                    ("Resume Alignment", f"{feedback['resume_alignment']}%"),
                    ("Keywords", f"{feedback['keyword_coverage']}%"),
                ]
            )
            st.write(feedback["feedback"])
            st.write("**Missing keywords to consider:** " + (", ".join(feedback["missing_keywords"]) or "None"))

with report_tab:
    if vectorstore is None:
        st.info("Process your resume to generate a downloadable prep report.")
    else:
        report = generate_prep_report(vectorstore, st.session_state.chat_history)
        st.download_button(
            "Download Interview Prep Report",
            data=report,
            file_name="interview_prep_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        with st.expander("Preview report"):
            st.markdown(report)

quick_prompts = [
    "Prepare me for a GenAI Developer interview with aptitude.",
    "Review my resume and tell me what to improve.",
    "Ask me mock interview questions based on my resume.",
    "Explain my RAG chatbot project for an interview.",
]

prompt_cols = st.columns(4)
for index, prompt in enumerate(quick_prompts):
    with prompt_cols[index]:
        if st.button(prompt, use_container_width=True):
            st.session_state.pending_question = prompt

with st.form("question_form", clear_on_submit=True):
    default_question = st.session_state.pending_question
    user_question = st.text_area(
        "Ask your interview question",
        value=default_question,
        height=110,
        placeholder="Example: I want to prepare for a GenAI Developer interview and aptitude. Please guide me.",
    )
    submitted = st.form_submit_button("Get Answer", type="primary")

if submitted:
    st.session_state.pending_question = ""
    ask_assistant(user_question)

last_answer = st.session_state.chat_history[-1]["answer"] if st.session_state.chat_history else ""

st.subheader("Voice Assistant")
voice_tools(last_answer)

st.subheader("Conversation")
if not st.session_state.chat_history:
    st.info("Upload and process your resume, then ask a question to begin.")
else:
    for chat in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

st.caption("Built with Streamlit, PyPDF, scikit-learn, local hybrid RAG, optional Ollama, and browser speech tools.")
