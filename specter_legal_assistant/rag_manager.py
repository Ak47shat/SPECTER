from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to community version if langchain_huggingface is not available
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from .config import settings
from .logger import logger
import logging
from langchain.schema import Document
from typing import List
import hashlib

# Set logging to DEBUG level to see all information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger.setLevel(logging.DEBUG)

class RAGManager:
    def __init__(self):
        # Using a more compatible embedding model with optimized settings
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Only keep necessary parameters
            }
        )
        # Load existing vector store
        try:
            vector_store_path = settings.VECTOR_STORE_PATH
            if not os.path.exists(vector_store_path):
                logger.warning(f"Vector store not found at {vector_store_path}, creating new one...")
                self._create_new_vector_store()
            else:
                self.vector_store = FAISS.load_local(
                    vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # We trust our own vector store files
                )
                logger.info("Successfully loaded existing vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            logger.info("Attempting to create new vector store...")
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        """Create a new vector store from legal documents."""
        try:
            from langchain.schema import Document
            import json
            
            # Load legal documents
            legal_docs_path = settings.LEGAL_DOCS_PATH
            documents = []
            
            if os.path.exists(legal_docs_path):
                for filename in os.listdir(legal_docs_path):
                    if filename.endswith('.jsonl'):
                        filepath = os.path.join(legal_docs_path, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    if 'text' in data:
                                        documents.append(Document(page_content=data['text']))
                                except json.JSONDecodeError:
                                    continue
            
            if documents:
                # Create vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # Save the vector store
                os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
                self.vector_store.save_local(settings.VECTOR_STORE_PATH)
                logger.info(f"Created new vector store with {len(documents)} documents")
            else:
                logger.warning("No legal documents found, creating empty vector store")
                # Create empty vector store
                self.vector_store = FAISS.from_texts(["No legal documents available"], self.embeddings)
                
        except Exception as e:
            logger.error(f"Error creating new vector store: {str(e)}")
            # Create minimal fallback vector store
            self.vector_store = FAISS.from_texts(["Legal assistance not available"], self.embeddings)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context for a query using semantic search."""
        try:
            # Get more candidates than needed for better coverage
            candidates = self.vector_store.similarity_search_with_score(query, k=k*2)
            logger.info(f"Retrieved {len(candidates)} candidate documents")
            
            # Process documents to filter out duplicates and irrelevant content
            seen_content = set()  # Track unique content
            processed_docs = []
            
            for doc, score in candidates:
                # Skip if we've seen this content before
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash in seen_content:
                    logger.debug(f"Skipping duplicate document with score {score}")
                    continue
                
                seen_content.add(content_hash)
                processed_docs.append((doc.page_content, score))
                logger.info(f"Document score: {score}")
                
                # Log raw document for debugging
                logger.info("-" * 80)
                logger.info(f"Document {len(processed_docs)} - Score: {score:.4f}")
                logger.info(f"Content: {doc.page_content}")
                logger.info("-" * 80)
            
            # Sort by similarity score and take top k
            processed_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, _ in processed_docs[:k]]
            
            # Log final selection
            logger.info("\nFinal processed documents:")
            for i, doc in enumerate(top_docs, 1):
                logger.info(f"Top Doc {i}:")
                logger.info(doc)
                logger.info("-" * 80)
            
            # Combine documents with clear separation
            context = "\n\n---\n\n".join(top_docs)
            
            logger.info(f"Retrieved {len(top_docs)} documents for query: {query}")
            logger.info(f"Final context length: {len(context)} characters")
            logger.info(f"Retrieved context length: {len(context)}")
            logger.info("Retrieved context for debugging:")
            logger.info("=" * 80)
            logger.info(context)
            logger.info("=" * 80)
            
            return context
            
        except Exception as e:
            logger.error(f"Error in get_relevant_context: {str(e)}", exc_info=True)
            return ""
    
    def update_vector_store_with_new_files(self):
        """Update the vector store with any new files in the legal_docs directory."""
        try:
            from langchain.schema import Document
            import json
            
            # Load legal documents
            legal_docs_path = settings.LEGAL_DOCS_PATH
            documents = []
            
            if os.path.exists(legal_docs_path):
                for filename in os.listdir(legal_docs_path):
                    if filename.endswith('.jsonl'):
                        filepath = os.path.join(legal_docs_path, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    if 'text' in data:
                                        documents.append(Document(page_content=data['text']))
                                except json.JSONDecodeError:
                                    continue
            
            if documents:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # Save the vector store
                os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
                self.vector_store.save_local(settings.VECTOR_STORE_PATH)
                logger.info(f"Updated vector store with {len(documents)} documents")
            else:
                logger.warning("No legal documents found for update")
                
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            raise

rag_manager = RAGManager() 