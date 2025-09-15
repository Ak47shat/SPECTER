import os
import uuid
from fpdf import FPDF
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize FAISS (for vector storage)
dimension = 384  # typical for sentence-transformers
index = faiss.IndexFlatL2(dimension)
model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------ Utilities ------------------ #

def split_message(text: str, max_length: int = 3000):
    """Split long responses into chunks for readability."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def truncate_response(response: str, max_length: int = 2000):
    """Truncate response to avoid overflow in UI."""
    return response[:max_length] + "..." if len(response) > max_length else response


# ------------------ Core AI Function ------------------ #

def ask_legal_response(question: str) -> str:
    """Ask Groq LLM for legal/contract related answers."""
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ------------------ Document Generators ------------------ #

def generate_fir_pdf(details: dict, output_dir: str = "generated_docs") -> str:
    """Generate FIR PDF."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fir_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(output_dir, filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "First Information Report (FIR)", ln=True, align="C")
    pdf.ln(10)

    for key, value in details.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")

    pdf.output(filepath)
    return filepath


def generate_rental_agreement(details: dict, output_dir: str = "generated_docs") -> str:
    """Generate Rental Agreement PDF."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"rental_agreement_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(output_dir, filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Rental Agreement", ln=True, align="C")
    pdf.ln(10)

    for key, value in details.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")

    pdf.output(filepath)
    return filepath


def generate_consumer_complaint(details: dict, output_dir: str = "generated_docs") -> str:
    """Generate Consumer Complaint PDF."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"consumer_complaint_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(output_dir, filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Consumer Complaint", ln=True, align="C")
    pdf.ln(10)

    for key, value in details.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")

    pdf.output(filepath)
    return filepath


# ------------------ Vector Store ------------------ #

def update_vector_store(docs: list[str]):
    """Update FAISS vector index with new documents."""
    embeddings = model.encode(docs)
    index.add(np.array(embeddings, dtype=np.float32))


def search_vector_store(query: str, top_k: int = 3):
    """Search in FAISS vector store."""
    embedding = model.encode([query])
    distances, indices = index.search(np.array(embedding, dtype=np.float32), top_k)
    return indices, distances
