import streamlit as st
import asyncio
from pathlib import Path
from specter_legal_assistant.rag_manager import rag_manager
from specter_legal_assistant.config import settings
from specter_legal_assistant.utils import format_response_for_gradio
from specter_legal_assistant import groq_client
from fpdf import FPDF
from datetime import datetime
import os
import uuid

# ---------- UTILITIES ---------- #

def generate_fir(name: str, location: str, details: str) -> str:
    """Generate FIR PDF with only user-provided details filled."""
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fir_{uuid.uuid4().hex}.pdf"
    filepath = Path(output_dir) / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "FIRST INFORMATION REPORT (FIR)", ln=True, align="C")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Police Station: _________________\n"
                           f"District: _________________\n"
                           f"State: _________________\n"
                           f"FIR No.: _________________\n"
                           f"Date of Registration: {datetime.today().strftime('%d-%m-%Y')}\n"
                           f"Time of Registration: {datetime.today().strftime('%H:%M:%S')}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"1. COMPLAINANT DETAILS\n"
                           f"Name: {name}\n"
                           f"Address: {location}\n"
                           f"Contact No.: _________________\n"
                           f"Email: _________________\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"2. INCIDENT DETAILS\n"
                           f"Date of Incident: _________________\n"
                           f"Time of Incident: _________________\n"
                           f"Place of Incident: {location}\n"
                           f"Detailed Description:\n{details}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, "3. SECTION OF LAW\nTo be filled by the Police\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, "4. WITNESS DETAILS\nName: _________________\nAddress: _________________\nContact: _________________\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, "5. PROPERTY DETAILS (IF ANY)\nDescription: _________________\nEstimated Value: _________________\n")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "6. SIGNATURES\n")
    pdf.cell(90, 10, "Complainant's Signature", border=0)
    pdf.cell(0, 10, "Police Officer's Signature", ln=True)
    pdf.cell(90, 10, "_________________", border=0)
    pdf.cell(0, 10, "_________________", ln=True)
    pdf.cell(90, 10, f"Name: {name}", border=0)
    pdf.cell(0, 10, "Name: _________________", ln=True)
    pdf.cell(90, 10, f"Date: {datetime.today().strftime('%d-%m-%Y')}", border=0)
    pdf.cell(0, 10, "Designation: _________________", ln=True)
    pdf.output(filepath)
    return str(filepath)

def generate_rental_agreement(landlord_name, tenant_name, property_address,
                              rent_amount, security_deposit, lease_start_date,
                              lease_end_date, terms_and_conditions) -> str:
    """Generate Rental Agreement PDF."""
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"rental_agreement_{uuid.uuid4().hex}.pdf"
    filepath = Path(output_dir) / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "RENTAL AGREEMENT", ln=True, align="C")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"1. PARTIES\nLandlord: {landlord_name}\nTenant: {tenant_name}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"2. PROPERTY DETAILS\nAddress: {property_address}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"3. LEASE TERMS\nRent Amount: INR {rent_amount:,.2f} per month\n"
                           f"Security Deposit: INR {security_deposit:,.2f}\n"
                           f"Lease Start Date: {lease_start_date}\n"
                           f"Lease End Date: {lease_end_date}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"4. TERMS AND CONDITIONS\n{terms_and_conditions}\n")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "5. SIGNATURES\n")
    pdf.cell(90, 10, "Landlord's Signature", border=0)
    pdf.cell(0, 10, "Tenant's Signature", ln=True)
    pdf.cell(90, 10, "_________________", border=0)
    pdf.cell(0, 10, "_________________", ln=True)
    pdf.cell(90, 10, f"Name: {landlord_name}", border=0)
    pdf.cell(0, 10, f"Name: {tenant_name}", ln=True)
    pdf.cell(90, 10, f"Date: {datetime.today().strftime('%d-%m-%Y')}", border=0)
    pdf.cell(0, 10, f"Date: {datetime.today().strftime('%d-%m-%Y')}", ln=True)
    pdf.output(filepath)
    return str(filepath)

def generate_consumer_complaint(complainant_name, complainant_address, complainant_contact,
                                company_name, company_address, product_service_details,
                                complaint_details, desired_resolution) -> str:
    """Generate Consumer Complaint PDF."""
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"consumer_complaint_{uuid.uuid4().hex}.pdf"
    filepath = Path(output_dir) / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "CONSUMER COMPLAINT", ln=True, align="C")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"1. COMPLAINANT DETAILS\nName: {complainant_name}\n"
                           f"Address: {complainant_address}\n"
                           f"Contact: {complainant_contact}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"2. COMPANY DETAILS\nCompany Name: {company_name}\n"
                           f"Address: {company_address}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"3. COMPLAINT DETAILS\nProduct/Service: {product_service_details}\n"
                           f"Complaint Description:\n{complaint_details}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"4. DESIRED RESOLUTION\n{desired_resolution}\n")
    pdf.ln(5)
    pdf.multi_cell(0, 10, "5. SIGNATURE\n")
    pdf.cell(0, 10, f"Complainant's Signature: _________________    Date: {datetime.today().strftime('%d-%m-%Y')}", ln=True)
    pdf.cell(0, 10, f"Name: {complainant_name}", ln=True)
    pdf.output(filepath)
    return str(filepath)

# ---------- LEGAL QUERY ---------- #
async def legal_query(query: str, language: str = "english") -> str:
    """Handle legal queries through RAG system"""
    try:
        if not query.strip():
            return "⚠️ Please enter a legal question."
        context = rag_manager.get_relevant_context(query, k=3)
        prompt = f"""You are a helpful legal assistant providing clear, practical advice about Indian law. 
Provide a natural, empathetic response that directly addresses their specific situation.

User's Question: {query}

Legal Information:
{context}

Provide a clear, practical response that explains their legal options and applicable laws in natural language.
"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant that provides clear, accurate information about Indian law."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        result = response.choices[0].message.content
        return format_response_for_gradio(result)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------- STREAMLIT UI ---------- #
st.set_page_config(page_title="Legal Assistant AI", layout="wide")

st.markdown(
    """
    <style>
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .stDownloadButton>button {
            background-color: #667eea;
            color: white;
            border-radius: 8px;
        }
        .stDownloadButton>button:hover {
            background-color: #764ba2;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
        <h1>🏛️ Legal Assistant AI</h1>
        <p>Get instant legal advice and generate legal documents under Indian law</p>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.title("📂 Menu")
app_mode = st.sidebar.radio(
    "Choose a tool:",
    [
        "🤖 Ask Legal Questions",
        "📝 Generate FIR",
        "🏠 Rental Agreement",
        "🛒 Consumer Complaint"
    ]
)

# ---------- Ask Legal Questions ----------
if app_mode == "🤖 Ask Legal Questions":
    st.subheader("🤖 Ask Legal Questions")
    query = st.text_area("Enter your legal question:", placeholder="Example: What are my rights if I'm arrested by the police?")
    language = st.selectbox("Select Language", ["english", "hindi"])
    if st.button("Get Legal Advice"):
        answer = asyncio.run(legal_query(query, language))
        st.text_area("Legal Advice", answer, height=300)

# ---------- FIR ----------
elif app_mode == "📝 Generate FIR":
    st.subheader("📝 FIR Generator")
    name = st.text_input("Your Full Name")
    location = st.text_input("Location of Incident")
    details = st.text_area("Incident Details")
    if st.button("Generate FIR"):
        file_path = generate_fir(name, location, details)
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download FIR", f, file_name="FIR.pdf")
        else:
            st.error(file_path)

# ---------- Rental Agreement ----------
elif app_mode == "🏠 Rental Agreement":
    st.subheader("🏠 Rental Agreement Generator")
    landlord = st.text_input("Landlord Name")
    tenant = st.text_input("Tenant Name")
    property_address = st.text_area("Property Address")
    rent = st.number_input("Monthly Rent (₹)", min_value=0.0)
    deposit = st.number_input("Security Deposit (₹)", min_value=0.0)
    lease_start = st.text_input("Lease Start Date (DD-MM-YYYY)")
    lease_end = st.text_input("Lease End Date (DD-MM-YYYY)")
    terms = st.text_area("Terms and Conditions")
    if st.button("Generate Rental Agreement"):
        file_path = generate_rental_agreement(landlord, tenant, property_address, rent, deposit, lease_start, lease_end, terms)
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download Rental Agreement", f, file_name="Rental_Agreement.pdf")
        else:
            st.error(file_path)

# ---------- Consumer Complaint ----------
elif app_mode == "🛒 Consumer Complaint":
    st.subheader("🛒 Consumer Complaint Generator")
    comp_name = st.text_input("Your Name")
    comp_address = st.text_area("Your Address")
    comp_contact = st.text_input("Your Contact Number")
    company = st.text_input("Company Name")
    company_addr = st.text_area("Company Address")
    product_details = st.text_input("Product/Service Details")
    complaint_details = st.text_area("Complaint Details")
    resolution = st.text_area("Desired Resolution")
    if st.button("Generate Complaint"):
        file_path = generate_consumer_complaint(comp_name, comp_address, comp_contact, company, company_addr,
                                                product_details, complaint_details, resolution)
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download Complaint", f, file_name="Consumer_Complaint.pdf")
        else:
            st.error(file_path)
