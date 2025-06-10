# Document Question Answering Web App

This project demonstrates how to build a simple website that lets you upload PDF, Word, or Excel files and ask questions about their contents using an open source language model.

## Requirements

- Python 3.11+
- Packages in `requirements.txt`

Install them with:

```bash
pip install -r requirements.txt
```

The first time you run this it will download open-source models (e.g., `flan-t5-small`).

## Running the Web App

```bash
python app.py
```

Then open `http://localhost:5000` in your browser. Upload documents and submit questions.

## Mobile App Idea

You can create a simple mobile app (React Native, Flutter, or any framework) that sends file uploads and questions to the same Flask backend. The backend responds with the answer in JSON or HTML.

For production, you might deploy the Flask server and call it from your mobile app via HTTP.

## Notes

- This is a minimal example for local experimentation. For large documents or heavy usage, consider using a more robust vector store and model hosting solution.
- Google NotebookLM is another option if you prefer a managed solution from Google.
