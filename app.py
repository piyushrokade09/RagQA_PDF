# from flask import Flask, request, jsonify, render_template
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# import os
# from groq import Groq
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend communication

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialize ChromaDB
# embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)

# # Groq API Client
# client = Groq(api_key="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")


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     # Extract text from PDF
#     reader = PdfReader(filepath)
#     book = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
#     texts = text_splitter.create_documents([book])
#     chunks = [d.page_content for d in texts]

#     # Store chunks in ChromaDB
#     ids = [str(i) for i in range(len(chunks))]
#     collection.add(documents=chunks, ids=ids)

#     return jsonify({"message": "File uploaded and processed successfully"})


# @app.route('/ask', methods=['POST'])
# def ask_question():
#     data = request.json
#     query = data.get("question", "")

#     if not query:
#         return jsonify({"error": "No question provided"}), 400

#     results = collection.query(query_texts=[query], n_results=3)
#     contexts = ' '.join(results['documents'][0]) if results['documents'] else ""

#     prompt = f"""Act as an AI assistant for a question-answering system.\n\n"""
#     prompt += f"Information: {contexts}\n"
#     prompt += f"Question: {query}\n\n"
#     prompt += "Provide an accurate answer. If no relevant information is found, say 'No information found in the given context'."

#     chat_completion = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile",
#     )

#     answer = chat_completion.choices[0].message.content
#     return jsonify({"answer": answer})


# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, request, jsonify, render_template
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# import os
# from groq import Groq
# from flask_cors import CORS
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# from concurrent.futures import ThreadPoolExecutor

# app = Flask(__name__)
# CORS(app)

# # Rate Limiting for Security
# limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialize ChromaDB with optimized settings
# embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)

# # Groq API Client (Use environment variable for security)
# client = Groq(api_key="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")

# executor = ThreadPoolExecutor()

# def process_pdf(filepath):
#     reader = PdfReader(filepath)
#     book = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
#     texts = text_splitter.create_documents([book])
#     chunks = [d.page_content for d in texts]
#     ids = [str(i) for i in range(len(chunks))]
#     collection.add(documents=chunks, ids=ids)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# @limiter.limit("5 per minute")  # Prevent abuse
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)
    
#     executor.submit(process_pdf, filepath)  # Process asynchronously
#     return jsonify({"message": "File uploaded. Processing in background."})

# @app.route('/ask', methods=['POST'])
# @limiter.limit("10 per minute")
# def ask_question():
#     data = request.json
#     query = data.get("question", "").strip()
#     if not query:
#         return jsonify({"error": "No question provided"}), 400
    
#     results = collection.query(query_texts=[query], n_results=3)
#     contexts = ' '.join(results['documents'][0]) if results['documents'] else ""
    
#     prompt = ("You are an AI assistant for a PDF question-answering system.\n"
#               f"Relevant Context: {contexts}\n"
#               f"Question: {query}\n"
#               "Provide a concise, accurate response. If unsure, say 'No relevant information found'.")
    
#     chat_completion = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile",
#     )
    
#     answer = chat_completion.choices[0].message.content
#     return jsonify({"answer": answer})

# if __name__ == '__main__':
#     app.run(debug=True)









import os
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Rate Limiting for Security
limiter = Limiter(get_remote_address, app=app, default_limits=["20 per minute"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB with optimized settings
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)

# Groq API Client (Secure API key usage)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

executor = ThreadPoolExecutor()

def process_pdf(filepath):
    """Extracts text from PDF, splits into chunks, and stores in ChromaDB."""
    try:
        reader = PdfReader(filepath)
        book = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.create_documents([book])
        chunks = [d.page_content for d in texts]
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
    except Exception as e:
        print(f"Error processing PDF: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")  # Prevent abuse
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    executor.submit(process_pdf, filepath)  # Process asynchronously
    return jsonify({"message": "File uploaded. Processing in background."})

@app.route('/ask', methods=['POST'])
@limiter.limit("10 per minute")
def ask_question():
    data = request.json
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400
    
    results = collection.query(query_texts=[query], n_results=3)
    contexts = ' '.join(results.get('documents', [[]])[0]) if results.get('documents') else "No relevant context found."
    
    prompt = ("You are an AI assistant for a PDF question-answering system.\n"
              f"Relevant Context: {contexts}\n"
              f"Question: {query}\n"
              "Provide a concise, accurate response. If unsure, say 'No relevant information found'.")
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    answer = chat_completion.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
