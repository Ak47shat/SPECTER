# SPECTER Legal Assistant

SPECTER is an AI-powered legal assistant for Indian law. It provides a Streamlit UI to:

- Ask legal questions with RAG-backed answers
- Generate FIR PDFs
- Generate Rental Agreements
- Generate Consumer Complaints

The app uses local legal documents for retrieval and a model served via Groq for generation.

## Project Structure
- `streamlit_app.py` – Streamlit UI (recommended entrypoint)
- `gradio_app.py` – Optional Gradio UI
- `specter_legal_assistant/` – Core logic, config, PDF generators, RAG
- `data/legal_docs/` – JSONL legal corpora used by RAG
- `static/` – Generated PDFs are written here
- `requirements.txt` – Python dependencies

## Prerequisites
- Python 3.11+ recommended
- A Groq API key for model inference

## Setup
1. Clone and enter the project
   ```bash
   git clone https://github.com/Ak47shat/SPECTER.git
   cd SPECTER
   ```
2. (Optional) Create a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file at the project root with at least:
   ```env
   # Required for AI responses
   GROQ_API_KEY=your_groq_api_key

   # Optional app settings (defaults shown)
   API_KEY=your_default_api_key
   LOG_LEVEL=INFO
   LEGAL_DOCS_PATH=data/legal_docs
   VECTOR_STORE_PATH=data/vector_store
   PDF_STORAGE_PATH=storage/pdfs
   PUBLIC_BASE_URL=http://localhost:8001
   ```

Note: Large vector indexes should not be committed. Ensure these patterns exist in `.gitignore`:
```
data/vector_store/
*.faiss
__pycache__/
*.pyc
```

## Run (Streamlit)
```bash
streamlit run streamlit_app.py
```

## Usage in the UI
- 🤖 Ask Legal Questions: Enter your question and pick language (english/hindi). The app retrieves top-k context from `data/legal_docs/` and generates a concise answer.
- 📝 Generate FIR: Fill name, location, and incident details; download a PDF.
- 🏠 Rental Agreement: Enter party details, property and lease terms; download a PDF.
- 🛒 Consumer Complaint: Provide complainant and company details; download a PDF.

Generated files are saved under `static/` for download.


## Notes
- If AI responses fail, verify `GROQ_API_KEY` is set and valid.
- You can adjust retrieval size and model in `specter_legal_assistant/config.py`.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
