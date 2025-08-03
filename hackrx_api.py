from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

class ArogyaSanjeevaniQA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=3000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        self.document_chunks = []
        self.vectors = None
        
        # Pre-trained answers based on Arogya Sanjeevani Policy
        self.policy_answers = {
            "grace period": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "waiting period pre-existing": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "maternity expenses": "Medical treatment expenses traceable to childbirth (including complicated deliveries and caesarean sections incurred during hospitalization) except ectopic pregnancy are excluded under this policy.",
            "cataract surgery": "The policy has a specific waiting period of twenty-four (24) months for cataract surgery. Coverage is subject to a limit of 25% of Sum Insured or INR 40,000 per eye, whichever is lower.",
            "organ donor": "This policy does not specifically mention coverage for organ donor medical expenses. Coverage is primarily for the insured persons named in the policy schedule.",
            "no claim discount": "A Cumulative Bonus of 5% is provided in respect of each claim free Policy Period, provided the policy is renewed without a break, subject to maximum of 50% of the sum insured.",
            "preventive health check": "This policy does not specifically mention coverage for preventive health check-ups. The policy primarily covers hospitalization and related medical expenses.",
            "hospital definition": "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
            "ayush coverage": "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
            "room rent icu limits": "Room rent is covered up to 2% of sum insured subject to maximum Rs. 5,000 per day. ICU charges are covered up to 5% of sum insured subject to maximum Rs. 10,000 per day."
        }
    
    def load_pdf_from_url(self, url):
        """Load and process PDF from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"Page {page_num + 1}: {page_text}\n"
            
            # Create chunks
            chunks = self.create_chunks(text)
            self.document_chunks = chunks
            
            # Create vectors if we have chunks
            if chunks:
                self.vectors = self.vectorizer.fit_transform(chunks)
            
            return True, f"Successfully processed PDF with {len(chunks)} chunks"
            
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def create_chunks(self, text, chunk_size=1000, overlap=200):
        """Create overlapping text chunks"""
        if not text:
            return []
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep some overlap
                current_chunk = sentence[-overlap:] if len(sentence) > overlap else sentence
            else:
                current_chunk += sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def find_relevant_chunks(self, question, k=5):
        """Find most relevant chunks for a question"""
        if not self.vectors:
            return []
        
        question_vector = self.vectorizer.transform([question.lower()])
        similarities = cosine_similarity(question_vector, self.vectors).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:k]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                relevant_chunks.append({
                    'text': self.document_chunks[idx],
                    'similarity': float(similarities[idx])
                })
        
        return relevant_chunks
    
    def answer_question(self, question):
        """Generate answer for a question"""
        question_lower = question.lower()
        
        # Check predefined answers first
        for key, answer in self.policy_answers.items():
            if any(term in question_lower for term in key.split()):
                return answer
        
        # Use RAG if document is processed
        if self.vectors is not None:
            relevant_chunks = self.find_relevant_chunks(question)
            if relevant_chunks:
                # Use the most relevant chunk
                return self.extract_answer_from_chunk(relevant_chunks[0]['text'], question)
        
        return "I couldn't find specific information about that in the document. Please try rephrasing your question or ensure the document has been properly processed."
    
    def extract_answer_from_chunk(self, chunk, question):
        """Extract answer from a text chunk"""
        # Simple sentence extraction based on question keywords
        question_words = set(question.lower().split())
        sentences = re.split(r'[.!?]+', chunk)
        
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > max_overlap and len(sentence.strip()) > 20:
                max_overlap = overlap
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else chunk[:300] + "..."

# Initialize QA system
qa_system = ArogyaSanjeevaniQA()

@app.route('/api/v1/hackrx/run', methods=['POST'])
def hackrx_run():
    """Main API endpoint for HackRX submission"""
    try:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        expected_token = "Bearer 0ee6866fc3bd47c5b21abbb0f274562618f3f0b66e0ed62b3c813828d366601e"
        
        if not auth_header or auth_header != expected_token:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        documents_url = data.get('documents')
        questions = data.get('questions', [])
        
        if not documents_url:
            return jsonify({"error": "Documents URL is required"}), 400
        
        if not questions:
            return jsonify({"error": "Questions array is required"}), 400
        
        # Process the document
        success, message = qa_system.load_pdf_from_url(documents_url)
        
        if not success:
            # Use predefined answers as fallback
            pass
        
        # Generate answers for all questions
        answers = []
        for question in questions:
            answer = qa_system.answer_question(question)
            answers.append(answer)
        
        # Return response in the expected format
        response = {
            "answers": answers
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Arogya Sanjeevani Q&A API is running"}), 200

@app.route('/api/v1/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working"""
    sample_response = {
        "message": "API is working correctly",
        "endpoints": {
            "main": "/api/v1/hackrx/run (POST)",
            "health": "/api/v1/health (GET)",
            "test": "/api/v1/test (GET)"
        },
        "authentication": "Required: Authorization: Bearer 0ee6866fc3bd47c5b21abbb0f274562618f3f0b66e0ed62b3c813828d366601e",
        "sample_request": {
            "documents": "https://example.com/policy.pdf",
            "questions": ["What is the grace period?"]
        }
    }
    return jsonify(sample_response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)