import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
OLLAMA_DEFAULT_MODEL = "mistral"

GENAI_TOPICS = [
    "LLM",
    "prompt engineering",
    "RAG",
    "embeddings",
    "vector",
    "chunking",
    "LangChain",
    "Streamlit",
    "Ollama",
    "Mistral",
    "Groq",
    "Python",
    "NLP",
    "agents",
    "evaluation",
    "SQL",
]


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    section: str = "General"


class LocalConversationChain:
    def __init__(self, vectorstore, llm_provider="auto", llm_model="", api_key=""):
        self.vectorstore = vectorstore
        self.chat_history = []
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key

    def invoke(self, inputs):
        question = inputs.get("question", "").strip()
        relevant_chunks = self.vectorstore.search(question)
        answer = build_answer(
            question=question,
            chunks=relevant_chunks,
            vectorstore=self.vectorstore,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            api_key=self.api_key,
        )

        self.chat_history.append({"question": question, "answer": answer})
        return {"answer": answer}


class LocalVectorStore:
    """Free hybrid RAG: TF-IDF retrieval + keyword overlap + resume section boosts."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [chunk.text for chunk in chunks]
        self.full_text = normalize_text(" ".join(self.texts))
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, k=6):
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.matrix).flatten()
        query_terms = set(tokenize(query))

        scored = []
        for index, chunk in enumerate(self.chunks):
            chunk_terms = set(tokenize(chunk.text))
            keyword_score = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            section_score = section_boost(query, chunk.section)
            score = float(tfidf_scores[index]) + keyword_score + section_score
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in scored[:k] if score > 0]

    def section_text(self, section_name):
        matches = [
            chunk.text
            for chunk in self.chunks
            if section_name.lower() in chunk.section.lower()
        ]
        return normalize_text(" ".join(matches))

    def source_text(self, source_name):
        matches = [
            chunk.text
            for chunk in self.chunks
            if source_name.lower() in chunk.source.lower()
        ]
        return normalize_text(" ".join(matches))


def load_pdf(file_path, source_label="Resume"):
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            documents.append(
                DocumentChunk(
                    text=text,
                    source=source_label,
                    page=page_number,
                    section=detect_section(text),
                )
            )

    return documents


def split_documents(documents, chunk_size=360, chunk_overlap=70):
    chunks = []

    for document in documents:
        section_blocks = split_by_resume_sections(document.text)

        for section, text in section_blocks:
            words = text.split()
            start = 0

            while start < len(words):
                end = start + chunk_size
                chunk_text = " ".join(words[start:end]).strip()

                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            text=chunk_text,
                            source=document.source,
                            page=document.page,
                            section=section or document.section,
                        )
                    )

                if end >= len(words):
                    break

                start = max(end - chunk_overlap, start + 1)

    return chunks


def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("Cannot create a vector store without document chunks.")
    return LocalVectorStore(chunks)


def create_conversation_chain(vectorstore, llm_provider="auto", llm_model="", api_key=""):
    return LocalConversationChain(
        vectorstore=vectorstore,
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
    )


def build_answer(question, chunks, vectorstore, llm_provider="auto", llm_model="", api_key=""):
    intent = detect_intent(question)
    context = build_context(chunks, vectorstore, intent)
    fallback = fallback_answer(question, chunks, vectorstore, intent)

    llm_answer = generate_with_llm(
        provider=llm_provider,
        model=llm_model,
        api_key=api_key,
        question=question,
        context=context,
        intent=intent,
    )

    if llm_answer:
        return llm_answer

    return fallback


def generate_with_llm(provider, model, api_key, question, context, intent):
    prompt = build_prompt(question, context, intent)
    provider = (provider or "auto").lower()

    if provider in ["groq", "auto"]:
        answer = call_groq(
            prompt=prompt,
            api_key=api_key or os.getenv("GROQ_API_KEY", ""),
            model=model or os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL),
        )
        if answer or provider == "groq":
            return answer

    if provider in ["gemini", "auto"]:
        answer = call_gemini(
            prompt=prompt,
            api_key=api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
            model=model or os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL),
        )
        if answer or provider == "gemini":
            return answer

    if provider in ["ollama", "auto"]:
        return call_ollama(
            prompt=prompt,
            model=model or os.getenv("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL),
        )

    return ""


def build_prompt(question, context, intent):
    return f"""
You are a practical AI Interview Preparation Assistant for a fresher or junior candidate.

Goal:
- Answer the user's question directly.
- Use the resume/job-description context.
- Do not dump raw resume chunks.
- Be specific to GenAI Developer, Data Science, Python, SQL, RAG, LLMs, projects, resume review, and aptitude preparation.
- If the user asks for interview preparation, give a clear preparation plan.
- If the user asks for resume review, give concrete improvements.
- If the user asks for aptitude, give formulas, examples, and practice questions.
- Keep the answer structured and useful.

Intent: {intent}

Retrieved context:
{context}

User question:
{question}

Answer with these sections when useful:
1. Direct answer
2. What to prepare
3. Resume-based points to say in interview
4. Practice questions or next steps
""".strip()


def call_groq(prompt, api_key, model):
    if not api_key:
        return ""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise, accurate interview coach. Use the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.25,
        "max_tokens": 900,
    }

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""


def call_gemini(prompt, api_key, model):
    if not api_key:
        return ""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 900,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            parts = data["candidates"][0]["content"]["parts"]
            return "\n".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""


def call_ollama(prompt, model):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.25,
            "num_predict": 900,
        },
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "").strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""


def build_context(chunks, vectorstore, intent):
    section_names = ["Professional Summary", "Technical Skills", "Experience", "Projects", "Education"]
    sections = []

    for section in section_names:
        text = vectorstore.section_text(section)
        if text:
            sections.append(f"{section}: {shorten(text, 1000)}")

    retrieved = [
        f"Page {chunk.page} | {chunk.section}: {shorten(chunk.text, 850)}"
        for chunk in chunks[:6]
    ]

    profile = "\n\n".join(sections)
    evidence = "\n\n".join(retrieved)
    topics = ", ".join(extract_matching_terms(vectorstore.full_text, GENAI_TOPICS))

    return f"""
Detected resume topics: {topics or "Not enough topics detected"}

Key resume sections:
{profile or "No section text detected."}

Most relevant retrieved evidence:
{evidence or "No strong retrieval match found."}

Answer intent:
{intent}
""".strip()


def fallback_answer(question, chunks, vectorstore, intent):
    if intent == "interview_plan":
        return genai_interview_plan(vectorstore)
    if intent == "resume_review":
        return resume_review(vectorstore)
    if intent == "aptitude":
        return aptitude_plan()
    if intent == "project_explain":
        return project_answer(chunks)
    if intent == "questions":
        return practice_questions(vectorstore)
    return direct_context_answer(chunks)


def genai_interview_plan(vectorstore):
    skills = extract_matching_terms(vectorstore.full_text, GENAI_TOPICS)
    skill_text = comma_list(skills) or "Python, RAG, LLMs, NLP, SQL, and AI projects"

    return f"""
## Direct Answer
Yes. For a GenAI Developer interview, prepare around your strongest resume areas: {skill_text}.

## 7-Day Plan
1. Python, OOP, APIs, JSON, exceptions, and Streamlit flow.
2. LLM basics: tokens, prompts, temperature, context window, hallucination.
3. RAG: PDF loading, chunking, embeddings, vector search, retrieval, prompt building.
4. Projects: AI Chatbot with RAG, AI Code Explainer, Resume Analyzer.
5. SQL and data: joins, group by, EDA, preprocessing, feature engineering.
6. Aptitude: percentages, ratios, averages, time-work, speed-distance-time, probability.
7. Mock interview: tell me about yourself, project deep dive, HR questions.

## Interview-Ready Introduction
I am an AI and Data Science graduate with hands-on experience in Python, SQL, LLMs, RAG, NLP, and Streamlit. I have built GenAI projects like an AI chatbot with RAG, an AI code explainer, and resume analysis workflows. During my internship, I worked on HR analytics, LMS data, EDA, preprocessing, and LLM-based automation. I am looking for a GenAI Developer role where I can build practical AI applications.
""".strip()


def resume_review(vectorstore):
    summary = extract_section_or_fallback(vectorstore, "Professional Summary")
    skills = extract_section_or_fallback(vectorstore, "Technical Skills")
    projects = extract_section_or_fallback(vectorstore, "Projects")
    experience = extract_section_or_fallback(vectorstore, "Experience")

    return f"""
## Resume Review

## Overall Verdict
Your resume is good for GenAI/Data Science fresher roles. It clearly shows Python, SQL, LLMs, RAG, NLP, Streamlit, internship experience, and deployed projects.

## Improvements
- Add measurable impact: response time, users, dataset size, documents processed, accuracy, or time saved.
- Put your strongest GenAI project first.
- Make every project bullet follow: problem -> what you built -> technology -> result.
- Add a compact "Core GenAI Stack" line: Python, Streamlit, RAG, embeddings, vector search, prompt engineering, Ollama/Groq.
- Reduce repeated words like "AI-powered" and replace them with technical implementation details.

## Better Summary
AI and Data Science graduate with hands-on experience building GenAI applications using Python, LLMs, RAG, NLP, Streamlit, and automation workflows. Built deployed projects including AI chatbots, resume analysis systems, and code explanation tools. Experienced in EDA, preprocessing, prompt engineering, retrieval-based answering, and practical AI solution development.

## Resume Evidence
**Summary:** {shorten(summary, 420)}

**Skills:** {shorten(skills, 420)}

**Experience:** {shorten(experience, 420)}

**Projects:** {shorten(projects, 520)}
""".strip()


def aptitude_plan():
    return """
## Aptitude Prep Plan

## Priority Topics
1. Percentages
2. Ratios and proportions
3. Averages
4. Profit and loss
5. Time and work
6. Speed, distance, and time
7. Simple and compound interest
8. Probability
9. Permutations and combinations
10. Data interpretation

## Daily Routine
- 30 minutes formulas and concepts
- 45 minutes timed practice
- 15 minutes wrong-answer review

## Practice Questions
1. A number is increased by 20% and then decreased by 20%. What is the net change?
2. If 6 people finish work in 12 days, how many days will 9 people take?
3. The average of 5 numbers is 24. If one number is 30, what is the average of the remaining 4?
4. A train travels 180 km in 3 hours. What is its speed?
5. If success probability is 0.7, what is failure probability?
""".strip()


def project_answer(chunks):
    if not chunks:
        return "I could not find project details in the uploaded resume."

    context = "\n".join(f"- {clean_sentence(chunk.text)}" for chunk in chunks[:3])
    return f"""
## Project Explanation

## Resume Context
{context}

## Interview Answer Format
I built this project to solve a practical problem using GenAI. The application takes user input or documents, processes the text, retrieves relevant context, and generates a useful answer. My main contribution was building the Python/Streamlit workflow, integrating retrieval logic, designing the prompt flow, and creating a simple user interface. This helped me understand RAG, prompt engineering, retrieval quality, and deployment.
""".strip()


def practice_questions(vectorstore):
    skills = extract_matching_terms(vectorstore.full_text, GENAI_TOPICS)
    skill_text = comma_list(skills) or "Python, SQL, RAG, LLMs, NLP, and Streamlit"

    return f"""
## Mock Interview Questions

Based on your resume skills: {skill_text}

1. Tell me about yourself.
2. Explain your AI Chatbot with RAG project.
3. What is RAG?
4. What are embeddings?
5. How do you reduce hallucination in an LLM app?
6. Difference between keyword search and vector search?
7. Explain chunk size and chunk overlap.
8. What did you do in your Associate Data Scientist internship?
9. Explain one EDA task you performed.
10. SQL: difference between WHERE and HAVING?
11. Python: explain list, tuple, dict, and set.
12. Aptitude: if a value increases by 25%, how do you reverse it?
""".strip()


def direct_context_answer(chunks):
    if not chunks:
        return (
            "I could not find that in the uploaded documents. Ask about your skills, "
            "projects, experience, education, GenAI prep, or aptitude prep."
        )

    points = [extract_best_sentence(chunk.text) for chunk in chunks[:4]]
    points = [point for point in points if point]

    return f"""
## Direct Answer
{chr(10).join(f"- {point}" for point in points)}

## Interview-Ready Version
Use the strongest point above, add one project example, and finish with what you learned or improved.
""".strip()


def detect_intent(question):
    q = question.lower()

    if any(word in q for word in ["resume review", "review my resume", "check my resume"]):
        return "resume_review"
    if "aptitude" in q and not any(word in q for word in ["gen ai", "generative ai", "developer"]):
        return "aptitude"
    if any(word in q for word in ["prepare", "interview", "gen ai", "generative ai", "developer"]):
        return "interview_plan"
    if any(word in q for word in ["question", "mock", "practice"]):
        return "questions"
    if any(word in q for word in ["project", "explain", "built", "chatbot", "rag"]):
        return "project_explain"

    return "context"


def split_by_resume_sections(text):
    headings = [
        "Professional Summary",
        "Technical Skills",
        "Experience",
        "Projects",
        "Education",
        "Certifications",
        "Strengths",
        "Languages",
    ]

    pattern = "|".join(re.escape(heading) for heading in headings)
    parts = re.split(f"({pattern})", text, flags=re.IGNORECASE)

    if len(parts) <= 1:
        return [(detect_section(text), text)]

    blocks = []
    current_section = "General"
    heading_set = {heading.lower() for heading in headings}

    for part in parts:
        clean = part.strip()
        if not clean:
            continue
        if clean.lower() in heading_set:
            current_section = clean.title()
        else:
            blocks.append((current_section, clean))

    return blocks


def detect_section(text):
    lowered = text.lower()
    for section in [
        "professional summary",
        "technical skills",
        "experience",
        "projects",
        "education",
        "certifications",
        "strengths",
        "languages",
    ]:
        if section in lowered:
            return section.title()
    return "General"


def section_boost(query, section):
    q = query.lower()
    s = section.lower()
    if "project" in q and "project" in s:
        return 0.4
    if "skill" in q and "skill" in s:
        return 0.4
    if "experience" in q and "experience" in s:
        return 0.4
    if "education" in q and "education" in s:
        return 0.4
    return 0.0


def extract_section_or_fallback(vectorstore, section):
    text = vectorstore.section_text(section)
    if text:
        return text
    chunks = vectorstore.search(section, k=2)
    return " ".join(chunk.text for chunk in chunks)


def extract_matching_terms(text, terms):
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def extract_best_sentence(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return shorten(text, 280)
    return shorten(clean_sentence(sentences[0]), 320)


def clean_sentence(text):
    text = normalize_text(text)
    text = text.replace("•", "; ")
    return text


def tokenize(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", text.lower())


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def comma_list(items):
    unique_items = []
    for item in items:
        if item not in unique_items:
            unique_items.append(item)

    if not unique_items:
        return ""
    if len(unique_items) == 1:
        return unique_items[0]
    return ", ".join(unique_items[:-1]) + f", and {unique_items[-1]}"


def shorten(text, max_chars=900):
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return f"{trimmed}..."


def get_profile_insights(vectorstore):
    skills = extract_matching_terms(vectorstore.full_text, GENAI_TOPICS)
    projects = vectorstore.section_text("Projects")
    experience = vectorstore.section_text("Experience")
    education = vectorstore.section_text("Education")

    return {
        "skills": comma_list(skills) or "Python, SQL, GenAI, RAG, and AI projects",
        "projects": shorten(projects, 700) or "No project section detected.",
        "experience": shorten(experience, 700) or "No experience section detected.",
        "education": shorten(education, 350) or "No education section detected.",
    }


def analyze_jd_match(vectorstore):
    resume_text = vectorstore.source_text("Resume")
    jd_text = vectorstore.source_text("Job Description")

    if not jd_text:
        return {
            "available": False,
            "score": 0,
            "matched": [],
            "missing": [],
            "summary": "Upload a job description PDF to unlock JD matching.",
        }

    resume_terms = set(tokenize(resume_text))
    jd_terms = set(tokenize(jd_text))
    useful_terms = {
        term
        for term in jd_terms
        if len(term) > 2 and term not in {"and", "the", "for", "with", "you", "are", "this", "that"}
    }

    matched = sorted(useful_terms & resume_terms)
    missing = sorted(useful_terms - resume_terms)
    score = round((len(matched) / max(len(useful_terms), 1)) * 100)

    return {
        "available": True,
        "score": score,
        "matched": matched[:18],
        "missing": missing[:18],
        "summary": (
            "Strong match. Focus on project depth and interview examples."
            if score >= 65
            else "Partial match. Add missing JD keywords where they truthfully fit."
        ),
    }


def score_answer(question, user_answer, vectorstore):
    answer_terms = set(tokenize(user_answer))
    question_terms = set(tokenize(question))
    resume_terms = set(tokenize(vectorstore.full_text))
    genai_terms = {term.lower() for term in GENAI_TOPICS}

    relevance = round((len(answer_terms & question_terms) / max(len(question_terms), 1)) * 100)
    resume_alignment = round((len(answer_terms & resume_terms) / max(len(answer_terms), 1)) * 100)
    keyword_coverage = round((len(answer_terms & genai_terms) / max(len(genai_terms), 1)) * 100)
    overall = round((relevance * 0.35) + (resume_alignment * 0.45) + (keyword_coverage * 0.2))

    missing_keywords = sorted((question_terms | genai_terms) - answer_terms)[:8]

    return {
        "overall": min(overall, 100),
        "relevance": min(relevance, 100),
        "resume_alignment": min(resume_alignment, 100),
        "keyword_coverage": min(keyword_coverage, 100),
        "missing_keywords": missing_keywords,
        "feedback": build_score_feedback(overall),
    }


def build_score_feedback(score):
    if score >= 75:
        return "Strong answer. Add one measurable result or project detail to make it sharper."
    if score >= 50:
        return "Decent answer. Make it more specific with resume evidence and technical keywords."
    return "Needs improvement. Use a clear structure: definition, project example, result, learning."


def generate_prep_report(vectorstore, chat_history=None):
    insights = get_profile_insights(vectorstore)
    jd = analyze_jd_match(vectorstore)
    chat_history = chat_history or []

    recent_questions = "\n".join(
        f"- {item['question']}" for item in chat_history[-8:]
    ) or "- No questions asked yet."

    jd_section = (
        f"""
## Job Description Match
- Match score: {jd['score']}%
- Summary: {jd['summary']}
- Matched keywords: {', '.join(jd['matched']) or 'None detected'}
- Missing keywords: {', '.join(jd['missing']) or 'None detected'}
""".strip()
        if jd["available"]
        else "## Job Description Match\nUpload a JD to generate role-specific matching."
    )

    return f"""
# AI Interview Preparation Report

## Candidate Snapshot
- Core skills: {insights['skills']}
- Education: {insights['education']}

## Experience Evidence
{insights['experience']}

## Project Evidence
{insights['projects']}

{jd_section}

## Recommended Prep Plan
1. Practice a 60-second self-introduction.
2. Prepare deep explanations for RAG, embeddings, chunking, hallucination control, and vector search.
3. Prepare one STAR-format story for internship work.
4. Prepare two project demos: AI Chatbot with RAG and AI Code Explainer.
5. Practice aptitude daily: percentages, ratios, averages, time-work, probability, and data interpretation.

## Recent Practice Questions
{recent_questions}
""".strip()
