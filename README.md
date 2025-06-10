# AI Assistant for Electronic Maintenance and Troubleshooting

This example demonstrates a private web system that lets you upload technical documents (PDF, Word, Excel) and ask questions about them. The goal is to quickly search manuals and diagrams instead of reading them manually.

The project uses open‑source models with [LangChain](https://github.com/langchain-ai/langchain) to build a small document QA service.

## Requirements

- Python 3.11+
- Packages listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Web App

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Upload one or more documents and then type questions about their contents.

Uploaded files are stored locally in the `uploads/` folder. A FAISS index is written to `faiss_index/` so your data is preserved if you restart the server.

## Mobile App Idea

A simple mobile client (React Native, Flutter, etc.) can send file uploads and questions to this Flask backend over HTTP. The server responds with answers in JSON or HTML so the same logic works on desktop or mobile.

For production deployments consider hosting the Flask app on a cloud service and adding HTTPS and authentication.

## Notes

This is a minimal proof of concept aimed at small‑scale use. For many users or large documents you may need more robust storage and a heavier language model.
