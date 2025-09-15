import gradio as gr
import asyncio
from datetime import datetime
from specter_legal_assistant.rag_manager import rag_manager
from specter_legal_assistant.config import settings
from specter_legal_assistant.utils import format_response_for_gradio
from pathlib import Path
import json

async def legal_query(query: str, language: str = "english") -> str:
    """Handle legal queries through RAG system"""
    try:
        if not query.strip():
            return "Please enter a legal question."
        
        # Get context from RAG system
        context = rag_manager.get_relevant_context(query, k=3)
        
        # Prepare prompt
        prompt = f"""You are a helpful legal assistant providing clear, practical advice about Indian law. Provide a natural, empathetic response that directly addresses their specific situation.\n\nUser's Question: {query}\n\nLegal Information:\n{context}\n\nProvide a clear, practical response that explains their legal options and applicable laws in natural language."""
        
        # Use Groq client for response generation
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
        return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

def generate_fir(name: str, location: str, details: str) -> str:
    """Generate FIR document"""
    try:
        from specter_legal_assistant import generate_fir_pdf
        filename = generate_fir_pdf(name, location, details)
        file_path = Path("static") / filename
        return str(file_path.absolute())
    except Exception as e:
        return f"Error generating FIR: {str(e)}"

def generate_rental_agreement(landlord_name: str, tenant_name: str, property_address: str, 
                            rent_amount: float, security_deposit: float, 
                            lease_start_date: str, lease_end_date: str, 
                            terms_and_conditions: str) -> str:
    """Generate Rental Agreement document"""
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
        file_path = Path("static") / filename
        return str(file_path.absolute())
    except Exception as e:
        return f"Error generating Rental Agreement: {str(e)}"

def generate_consumer_complaint(complainant_name: str, complainant_address: str,
                             complainant_contact: str, company_name: str,
                             company_address: str, product_service_details: str,
                             complaint_details: str, desired_resolution: str) -> str:
    """Generate Consumer Complaint document"""
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
        file_path = Path("static") / filename
        return str(file_path.absolute())
    except Exception as e:
        return f"Error generating Consumer Complaint: {str(e)}"

def create_gradio_interface():
    # Create the Gradio interface with improved design
    with gr.Blocks(
        title="Legal Assistant AI", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏛️ Legal Assistant AI</h1>
            <p>Get instant legal advice and generate legal documents for Indian law</p>
        </div>
        """)
        
        with gr.Tabs():
            # Legal Query Tab
            with gr.Tab("🤖 Ask Legal Questions"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Enter your legal question", 
                            lines=4,
                            placeholder="Example: What are my rights if I'm arrested by the police?"
                        )
                        language_dropdown = gr.Dropdown(
                            choices=["english", "hindi"], 
                            value="english",
                            label="Language"
                        )
                        query_button = gr.Button("Get Legal Advice", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Tips for better responses:
                        - Be specific about your situation
                        - Mention relevant details (location, circumstances)
                        - Ask about specific laws or procedures
                        - Include any relevant dates or events
                        """)
                
                query_output = gr.Textbox(
                    label="Legal Advice", 
                    lines=8,
                    interactive=False,
                    show_copy_button=True
                )
                
                query_button.click(
                    fn=lambda q, l: asyncio.run(legal_query(q, l)),
                    inputs=[query_input, language_dropdown],
                    outputs=query_output
                )

            # FIR Generation Tab
            with gr.Tab("📝 Generate FIR"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### First Information Report (FIR) Generator")
                        fir_name = gr.Textbox(
                            label="Your Full Name", 
                            placeholder="Enter your complete name"
                        )
                        fir_location = gr.Textbox(
                            label="Location of Incident", 
                            placeholder="Where did the incident occur?"
                        )
                        fir_details = gr.Textbox(
                            label="Incident Details", 
                            lines=4,
                            placeholder="Describe what happened in detail..."
                        )
                        fir_button = gr.Button("Generate FIR Document", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📋 FIR Information:
                        - Used to report criminal offenses
                        - Filed at the nearest police station
                        - Required for legal proceedings
                        - Contains complainant and incident details
                        """)
                
                fir_output = gr.File(label="Download Generated FIR", file_count="single")
                fir_button.click(
                    fn=generate_fir,
                    inputs=[fir_name, fir_location, fir_details],
                    outputs=fir_output
                )

            # Rental Agreement Tab
            with gr.Tab("🏠 Rental Agreement"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Rental Agreement Generator")
                        landlord_name = gr.Textbox(label="Landlord Name", placeholder="Enter landlord's full name")
                        tenant_name = gr.Textbox(label="Tenant Name", placeholder="Enter tenant's full name")
                        property_address = gr.Textbox(
                            label="Property Address", 
                            lines=2,
                            placeholder="Complete address of the rental property"
                        )
                        rent_amount = gr.Number(label="Monthly Rent Amount (₹)", precision=2)
                        security_deposit = gr.Number(label="Security Deposit Amount (₹)", precision=2)
                        lease_start = gr.Textbox(label="Lease Start Date (DD-MM-YYYY)", placeholder="01-01-2024")
                        lease_end = gr.Textbox(label="Lease End Date (DD-MM-YYYY)", placeholder="31-12-2024")
                        terms = gr.Textbox(
                            label="Terms and Conditions", 
                            lines=3,
                            placeholder="Additional terms and conditions..."
                        )
                        rental_button = gr.Button("Generate Rental Agreement", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📄 Rental Agreement Info:
                        - Legal contract between landlord and tenant
                        - Defines rent, deposit, and lease terms
                        - Protects both parties' rights
                        - Required for rental disputes
                        """)
                
                rental_output = gr.File(label="Download Rental Agreement", file_count="single")
                rental_button.click(
                    fn=generate_rental_agreement,
                    inputs=[landlord_name, tenant_name, property_address,
                           rent_amount, security_deposit, lease_start,
                           lease_end, terms],
                    outputs=rental_output
                )

            # Consumer Complaint Tab
            with gr.Tab("🛒 Consumer Complaint"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Consumer Complaint Generator")
                        comp_name = gr.Textbox(label="Your Name", placeholder="Enter your full name")
                        comp_address = gr.Textbox(
                            label="Your Address", 
                            lines=2,
                            placeholder="Your complete address"
                        )
                        comp_contact = gr.Textbox(label="Your Contact Number", placeholder="+91-XXXXXXXXXX")
                        company_name = gr.Textbox(label="Company Name", placeholder="Name of the company you're complaining against")
                        company_addr = gr.Textbox(
                            label="Company Address", 
                            lines=2,
                            placeholder="Company's address"
                        )
                        product_details = gr.Textbox(label="Product/Service Details", placeholder="What product or service are you complaining about?")
                        complaint_details = gr.Textbox(
                            label="Complaint Details", 
                            lines=3,
                            placeholder="Describe your complaint in detail..."
                        )
                        resolution = gr.Textbox(
                            label="Desired Resolution", 
                            lines=2,
                            placeholder="What resolution do you want?"
                        )
                        complaint_button = gr.Button("Generate Consumer Complaint", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 🛡️ Consumer Rights:
                        - Right to safety and quality
                        - Right to information
                        - Right to choose
                        - Right to redressal
                        - Right to be heard
                        """)
                
                complaint_output = gr.File(label="Download Consumer Complaint", file_count="single")
                complaint_button.click(
                    fn=generate_consumer_complaint,
                    inputs=[comp_name, comp_address, comp_contact,
                           company_name, company_addr, product_details,
                           complaint_details, resolution],
                    outputs=complaint_output
                )

    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name=settings.GRADIO_SERVER_NAME,
        server_port=settings.GRADIO_SERVER_PORT,
        share=settings.GRADIO_SHARE
    )
