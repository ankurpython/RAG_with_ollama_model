# RAG with Ollama Models

A simple implementation of Retrieval-Augmented Generation (RAG) using Ollama with TinyLlama and Llama3.2 models. Query your own documents using AI locally on your computer.

## What is this?

This project lets you:
- Upload your own documents (PDF, text files, etc.)
- Ask questions about those documents
- Get AI-powered answers using local models (no internet required for inference)

## Requirements

- Python 3.8+
- Ollama installed on your system

## Quick Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/ankurpython/RAG_with_ollama_model.git
   cd RAG_with_ollama_model
   ```

2. **Install Python packages**
   ```bash
   pip install langchain langchain-community sentence-transformers faiss-cpu
   ```

3. **Install Ollama**
   - Visit [ollama.ai](https://ollama.ai) and download for your OS
   - Or use: `curl -fsSL https://ollama.ai/install.sh | sh` (Linux/Mac)

4. **Download models**
   ```bash
   ollama pull tinyllama
   ollama pull llama3.2
   ```

## How to Use

1. **Put your documents** in the `data/` folder (create it if it doesn't exist)

2. **Run the main script**
   ```bash
   python main.py
   ```

3. **Ask questions** about your documents!

## Example Code

```python
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(documents=your_documents, embedding=embeddings)

# Set up the model
llm = Ollama(model="tinyllama")  # or "llama3.2"

# Ask a question
query = "What is the main topic discussed?"
relevant_docs = db.similarity_search(query)
response = llm.invoke(f"Based on this context: {relevant_docs}, answer: {query}")
print(response)
```

## Models

- **tinyllama**: Faster, smaller, good for testing
- **llama3.2**: Slower, larger, better quality answers


## Common Issues

**"ollama not found"**: Make sure Ollama is installed and running
**"Model not found"**: Run `ollama pull tinyllama` and `ollama pull llama3.2`
**Slow performance**: Try using tinyllama instead of llama3.2
