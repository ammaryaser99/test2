import os
from flask import Flask, request, render_template, redirect

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import pandas as pd
from pypdf import PdfReader
from docx import Document

app = Flask(__name__)

index_dir = "faiss_index"
vector_store = None

# Initialize models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-small",
    task="text2text-generation",
    device=0 if os.environ.get("USE_GPU") else -1,
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load existing index if it exists
if os.path.exists(index_dir):
    vector_store = FAISS.load_local(index_dir, embeddings)


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext in [".doc", ".docx"]:
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, sheet_name=None)
        return "\n".join(df[s].to_string() for s in df)
    else:
        with open(file_path, "r", errors="ignore") as f:
            return f.read()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global vector_store
    texts = []
    os.makedirs("uploads", exist_ok=True)
    for f in request.files.getlist("files"):
        path = os.path.join("uploads", f.filename)
        f.save(path)
        text = extract_text(path)
        docs = text_splitter.create_documents([text])
        texts.extend(docs)
    if texts:
        if vector_store is None:
            vector_store = FAISS.from_documents(texts, embeddings)
        else:
            vector_store.add_documents(texts)
        vector_store.save_local(index_dir)
    return redirect("/")


@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    if vector_store is None:
        return render_template("index.html", answer="No documents uploaded yet")
    question = request.form.get("question")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    answer = qa.run(question)
    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
