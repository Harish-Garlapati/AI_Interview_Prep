# AI_Interview_Prep

# AI Interview Preparation Assistant

A Streamlit app that lets you upload a resume PDF and an optional job description PDF, then ask interview-preparation questions using a fully local retrieval pipeline.

This version uses free/local tools. It can answer with a fast free-tier LLM provider or a local Ollama model. If no LLM is configured, it still falls back to the built-in offline interview coach.

## Features

- Free local hybrid RAG using TF-IDF, keyword matching, and resume section boosts.
- GenAI Developer interview preparation plan.
- Resume review with improvement suggestions.
- Aptitude preparation plan and practice questions.
- Mock interview questions based on uploaded resume content.
- Browser voice input and text-to-speech where supported.
- Fast LLM answers using Groq free tier, Gemini free tier, or local Ollama.
- Prep dashboard with candidate snapshot and one-click preparation actions.
- JD matching with matched and missing keywords when a job description is uploaded.
- Mock interview mode with answer scoring and improvement feedback.
- Downloadable Markdown interview preparation report.

## Setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL that Streamlit prints in the terminal.

## Deploy On Streamlit Cloud

Recommended setup:

1. Push this project to GitHub.
2. Go to Streamlit Cloud and create a new app from the GitHub repo.
3. Set the main file path as:

```text
app.py
```

4. Open the app settings, then add secrets.

For Groq:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

For Gemini:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-2.0-flash"
```

After this, users will not need to enter API keys in the sidebar. The deployed app reads the key from Streamlit Secrets automatically.

Do not commit your real `.env` file or API keys to GitHub.

## LLM Options

### Groq Free Tier

Use this for the fastest answers.

1. Create a free Groq API key.
2. In the app sidebar, choose `Groq free tier - fastest`.
3. Paste your key and process the resume.

Default model:

```text
llama-3.1-8b-instant
```

### Gemini Free Tier

1. Create a Gemini API key.
2. In the app sidebar, choose `Gemini free tier`.
3. Paste your key and process the resume.

Default model:

```text
gemini-2.0-flash
```

### Ollama Local

For more natural answers without paid APIs, install Ollama and run a local model:

```powershell
ollama pull mistral
ollama run mistral
```

The app will automatically use `mistral` through Ollama when it is available. Without Ollama, it still works using the built-in local RAG coach.
