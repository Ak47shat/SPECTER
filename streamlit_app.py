import streamlit as st
import asyncio
from pathlib import Path
from specter_legal_assistant.rag_manager import rag_manager
from specter_legal_assistant.config import settings
from specter_legal_assistant.utils import format_response_for_gradio


# ========== LEGAL QUERY ==========
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
        from specter_legal_assistant import groq_client
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


# ========== DOCUMENT GENERATORS ==========
def generate_fir(name: str, location: str, details: str) -> str:
    try:
        from specter_legal_assistant import generate_fir_pdf
        filename = generate_fir_pdf(name, location, details)
        return str(Path("static") / filename)
    except Exception as e:
        return f"Error generating FIR: {str(e)}"


def generate_rental_agreement(landlord_name, tenant_name, property_address,
                              rent_amount, security_deposit, lease_start_date,
                              lease_end_date, terms_and_conditions) -> str:
    try:
        from specter_legal_assistant import RentalAgreementData, generate_rental_agreement
        data = RentalAgreementData(
            landlord_name=landlord_name,
            tenant_name=tenant_name,
            property_address=property_address,
            rent_amount=rent_amount,
            security_deposit=security_deposit,
            lease_start_date=lease_start_date,
            lease_end_date=lease_end_date,
            terms_and_conditions=terms_and_conditions
        )
        filename = generate_rental_agreement(data)
        return str(Path("static") / filename)
    except Exception as e:
        return f"Error generating Rental Agreement: {str(e)}"


def generate_consumer_complaint(complainant_name, complainant_address, complainant_contact,
                                company_name, company_address, product_service_details,
                                complaint_details, desired_resolution) -> str:
    try:
        from specter_legal_assistant import ConsumerComplaintData, generate_consumer_complaint
        data = ConsumerComplaintData(
            complainant_name=complainant_name,
            complainant_address=complainant_address,
            complainant_contact=complainant_contact,
            company_name=company_name,
            company_address=company_address,
            product_service_details=product_service_details,
            complaint_details=complaint_details,
            desired_resolution=desired_resolution
        )
        filename = generate_consumer_complaint(data)
        return str(Path("static") / filename)
    except Exception as e:
        return f"Error generating Consumer Complaint: {str(e)}"


# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Legal Assistant AI", layout="wide")

# Custom CSS
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
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🏛️ Legal Assistant AI</h1>
        <p>Get instant legal advice and generate legal documents under Indian law</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
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


# -------- Ask Legal Questions --------
if app_mode == "🤖 Ask Legal Questions":
    st.subheader("🤖 Ask Legal Questions")
    query = st.text_area("Enter your legal question:", placeholder="Example: What are my rights if I'm arrested by the police?")
    language = st.selectbox("Select Language", ["english", "hindi"])
    
    with st.expander("💡 Tips for better responses"):
        st.markdown("""
        - Be specific about your situation  
        - Mention relevant details (location, circumstances)  
        - Ask about specific laws or procedures  
        - Include any relevant dates or events  
        """)
    
    if st.button("Get Legal Advice"):
        answer = asyncio.run(legal_query(query, language))
        st.text_area("Legal Advice", answer, height=300)


# -------- FIR --------
elif app_mode == "📝 Generate FIR":
    st.subheader("📝 FIR Generator")
    name = st.text_input("Your Full Name")
    location = st.text_input("Location of Incident")
    details = st.text_area("Incident Details")
    
    with st.expander("📋 FIR Information"):
        st.markdown("""
        - Used to report criminal offenses  
        - Filed at the nearest police station  
        - Required for legal proceedings  
        - Contains complainant and incident details  
        """)
    
    if st.button("Generate FIR"):
        file_path = generate_fir(name, location, details)
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download FIR", f, file_name="FIR.pdf")
        else:
            st.error(file_path)


# -------- Rental Agreement --------
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
    
    with st.expander("📄 Rental Agreement Info"):
        st.markdown("""
        - Legal contract between landlord and tenant  
        - Defines rent, deposit, and lease terms  
        - Protects both parties' rights  
        - Required for rental disputes  
        """)
    
    if st.button("Generate Rental Agreement"):
        file_path = generate_rental_agreement(landlord, tenant, property_address, rent, deposit, lease_start, lease_end, terms)
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download Rental Agreement", f, file_name="Rental_Agreement.pdf")
        else:
            st.error(file_path)


# -------- Consumer Complaint --------
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
    
    with st.expander("🛡️ Consumer Rights"):
        st.markdown("""
        - Right to safety and quality  
        - Right to information  
        - Right to choose  
        - Right to redressal  
        - Right to be heard  
        """)
    
    if st.button("Generate Complaint"):
        file_path = generate_consumer_complaint(comp_name, comp_address, comp_contact, company, company_addr,
                                                product_details, complaint_details, resolution)
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download Complaint", f, file_name="Consumer_Complaint.pdf")
        else:
            st.error(file_path)
