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








#
# import os
# from flask import Flask, request, jsonify, render_template
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from groq import Groq
# from flask_cors import CORS
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# from concurrent.futures import ThreadPoolExecutor
# from dotenv import load_dotenv
# import uuid
#
# # Load environment variables
# load_dotenv()
#
# app = Flask(__name__)
# CORS(app)
#
# # Rate Limiting for Security
# limiter = Limiter(get_remote_address, app=app, default_limits=["20 per minute"])
#
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # Initialize ChromaDB with optimized settings
# embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)
#
# # Groq API Client (Secure API key usage)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=GROQ_API_KEY)
#
# executor = ThreadPoolExecutor()
#
# def process_pdf(filepath):
#     """Extracts text from PDF, splits into chunks, and stores in ChromaDB."""
#     try:
#         reader = PdfReader(filepath)
#         book = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
#         texts = text_splitter.create_documents([book])
#         chunks = [d.page_content for d in texts]
#         ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
#         collection.add(documents=chunks, ids=ids)
#     except Exception as e:
#         print(f"Error processing PDF: {e}")
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route('/upload', methods=['POST'])
# @limiter.limit("5 per minute")  # Prevent abuse
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)
#
#     executor.submit(process_pdf, filepath)  # Process asynchronously
#     return jsonify({"message": "File uploaded. Processing in background."})
#
# @app.route('/ask', methods=['POST'])
# @limiter.limit("10 per minute")
# def ask_question():
#     data = request.json
#     query = data.get("question", "").strip()
#     if not query:
#         return jsonify({"error": "No question provided"}), 400
#
#     results = collection.query(query_texts=[query], n_results=3)
#     contexts = ' '.join(results.get('documents', [[]])[0]) if results.get('documents') else "No relevant context found."
#
#     prompt = ("You are an AI assistant for a PDF question-answering system.\n"
#               f"Relevant Context: {contexts}\n"
#               f"Question: {query}\n"
#               "Provide a concise, accurate response. If unsure, say 'No relevant information found'.")
#
#     chat_completion = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile",
#     )
#
#     answer = chat_completion.choices[0].message.content
#     return jsonify({"answer": answer})
#
# if __name__ == '__main__':
#     app.run(debug=True)


#  source /Users/apple/Desktop/rag/qaenv/bin/activate

#
# import os
# import uuid
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from groq import Groq
# from redis import Redis
# from concurrent.futures import ThreadPoolExecutor
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# app = Flask(__name__)
# CORS(app)
#
# # Configure Redis for rate limiting
# redis_client = Redis(host="localhost", port=6379, db=0)
# limiter = Limiter(get_remote_address, app=app, storage_uri="redis://localhost:6379")
#
# # Upload folder setup
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # Initialize ChromaDB with Sentence Transformer embeddings
# embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)
#
# # Groq API Client (Secure API key usage)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=GROQ_API_KEY)
#
# # Thread pool for async tasks
# executor = ThreadPoolExecutor()
#
#
# def process_pdf(filepath):
#     """Extracts text from PDF, splits into chunks, and stores in ChromaDB."""
#     try:
#         reader = PdfReader(filepath)
#         book = "".join([page.extract_text() or "" for page in reader.pages])
#
#         if not book.strip():
#             raise ValueError("PDF contains no extractable text.")
#
#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
#         texts = text_splitter.create_documents([book])
#
#         if not texts:
#             raise ValueError("Text splitting failed.")
#
#         # Store in ChromaDB
#         chunks = [d.page_content for d in texts]
#         ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
#         collection.add(documents=chunks, ids=ids)
#
#         print(f"Successfully processed and stored {len(chunks)} chunks in ChromaDB.")
#
#     except Exception as e:
#         print(f"Error processing PDF: {e}")
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# @limiter.limit("20 per minute")  # Adjusted for better testing
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)
#
#     executor.submit(process_pdf, filepath)  # Process asynchronously
#     return jsonify({"message": "File uploaded. Processing in background."})
#
#
# @app.route('/ask', methods=['POST'])
# @limiter.limit("20 per minute")
# def ask_question():
#     data = request.json
#     query = data.get("question", "").strip()
#     if not query:
#         return jsonify({"error": "No question provided"}), 400
#
#     # Retrieve relevant context from ChromaDB
#     query_results = collection.query(query_texts=[query], n_results=3)
#     contexts = " ".join(query_results.get("documents", [[]])[0]) if query_results.get(
#         "documents") else "No relevant context found."
#
#     prompt = (
#         "You are an AI assistant for a PDF question-answering system.\n"
#         f"Relevant Context: {contexts}\n"
#         f"Question: {query}\n"
#         "Provide a concise, accurate response. If unsure, say 'No relevant information found'."
#     )
#
#     chat_completion = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile",
#     )
#
#     answer = chat_completion.choices[0].message.content
#     return jsonify({"answer": answer})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

















import os
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq
from redis import Redis
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Redis for rate limiting
redis_client = Redis(host="localhost", port=6379, db=0)
limiter = Limiter(get_remote_address, app=app, storage_uri="redis://localhost:6379")

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB with Sentence Transformer embeddings
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_collection", embedding_function=embedding_function)

# Groq API Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Thread pool for async tasks
executor = ThreadPoolExecutor()


def process_pdf(filepath, filename):
    """Extracts text from PDF, splits into chunks, and stores in ChromaDB."""
    try:
        reader = PdfReader(filepath)
        book = "".join([page.extract_text() or "" for page in reader.pages])

        if not book.strip():
            raise ValueError("PDF contains no extractable text.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.create_documents([book])

        if not texts:
            raise ValueError("Text splitting failed.")

        # Store in ChromaDB
        chunks = [d.page_content for d in texts]
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadata = [{"filename": filename} for _ in range(len(chunks))]  # Store filename in metadata
        collection.add(documents=chunks, ids=ids, metadatas=metadata)

        print(f"Successfully processed and stored {len(chunks)} chunks in ChromaDB.")

    except Exception as e:
        print(f"Error processing PDF: {e}")


@app.route('/')
def home():
    pdf_files = os.listdir(UPLOAD_FOLDER)  # Get list of uploaded PDFs
    return render_template('index.html', pdf_files=pdf_files)


@app.route('/upload', methods=['POST'])
@limiter.limit("20 per minute")
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    executor.submit(process_pdf, filepath, file.filename)  # Process asynchronously
    return jsonify({"message": "File uploaded. Processing in background."})


@app.route('/delete', methods=['POST'])
def delete_pdf():
    data = request.json
    filename = data.get("filename", "")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Step 1: Remove the actual file from the upload folder
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted file: {filepath}")
    else:
        return jsonify({"error": "File not found"}), 404

    # Step 2: Remove associated data from ChromaDB
    try:
        # Get stored documents that match the filename
        query_results = collection.get()

        ids_to_delete = [
            doc_id for doc_id, meta in zip(query_results["ids"], query_results["metadatas"])
            if meta.get("filename") == filename
        ]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks from ChromaDB for {filename}.")
        else:
            print(f"No data found in ChromaDB for {filename}.")

    except Exception as e:
        print(f"Error deleting from ChromaDB: {e}")
        return jsonify({"error": "Failed to delete data from ChromaDB"}), 500

    return jsonify({"message": f"File '{filename}' and associated data deleted successfully."})


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Search for relevant text in ChromaDB
    results = collection.query(query_texts=[question], n_results=3)

    relevant_texts = [doc for doc_list in results["documents"] for doc in doc_list]

    if not relevant_texts:
        return jsonify({"answer": "No relevant information found in PDFs."})

    # Use Groq LLM to generate response
    context = " ".join(relevant_texts[:3])
    prompt = f"Answer the question based on the following content:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )

    answer = response.choices[0].message.content.strip()
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(debug=True)
