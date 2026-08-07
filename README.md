# 🤖 AI Document Research Assistant

An agentic document research assistant built for the GDG GenAI Recruitment Task. It answers user questions using a provided PDF document, and — only with explicit user permission — falls back to a live web search when the document doesn't contain enough information to answer.

## Problem It Solves

Answering questions from a document corpus without hallucinating, while giving the user full control over when the system is allowed to go beyond the documents and search the web.

## Features

- **PDF ingestion** — upload a PDF, which is parsed, chunked, and embedded into a vector store.
- **Retrieval-augmented answers** — every query is answered strictly from the retrieved document context, not the model's own knowledge.
- **Insufficient-context detection** — if the retrieved chunks don't support an answer, the assistant explicitly says so instead of guessing.
- **User-approved web search** — the assistant asks for permission before searching the web, and only proceeds if the user clicks "Yes."
- **Transparent retrieval** — retrieved chunks and similarity scores are shown in the UI, along with the source of the final answer (document vs. web).

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (`gemini-3.5-flash` via `langchain-google-genai`) |
| Embeddings | Google Generative AI Embeddings (`gemini-embedding-001`) |
| Vector store | ChromaDB (`langchain-chroma`) |
| PDF parsing | `PyPDFLoader` (`langchain-community`) |
| Web search | Tavily API |
| Orchestration | LangChain |

## Project Structure

```
.
├── streamlit_app.py     # Streamlit UI — file upload, query input, web-search consent flow
├── app.py                # Core logic — vector DB creation, retrieval, answer generation, web search
├── data/                  # Uploaded PDFs are stored here
├── database/              # Chroma persistent vector store
├── requirements.txt
├── .env                   # API keys (not committed)
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_google_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

5. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

## How to Use

1. Upload a PDF document using the file uploader.
2. Wait for it to be processed into the vector store.
3. Type a question in the text box and click **Ask**.
4. If the answer is found in the document, it's shown directly with the retrieved supporting chunks.
5. If the document doesn't contain enough information, the assistant will ask:
   *"I couldn't find sufficient information in the provided documents. Would you like me to search the web for this query?"*
   - Click **Yes** to allow a live web search and get a grounded answer from web sources.
   - Click **No** to decline — the assistant will not search the web.

## Known Limitations / Future Improvements

See the accompanying write-up (`writeup.md`) for a detailed discussion of what would be improved with more time — including more robust chunking, multi-PDF support, better rate-limit handling, and a more thorough self-correction loop.

## Demo Video
# AI-Document-Research-Assistant
