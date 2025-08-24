RAG with Ollama: TinyLlama and Llama3.2 Model Testing
This repository demonstrates the implementation of Retrieval-Augmented Generation (RAG) using the Ollama framework with the tinyllama and llama3.2 models. The project focuses on testing RAG workflows with custom datasets, combining retrieval-based methods with generative AI to produce contextually relevant responses.
Table of Contents

Project Overview
Features
Prerequisites
Installation
Usage


Project Overview
Retrieval-Augmented Generation (RAG) enhances language models by retrieving relevant documents from a custom dataset before generating responses. This project uses Ollama to run tinyllama and llama3.2 models locally, integrating them with a RAG pipeline to process and query custom data. It serves as a proof-of-concept for building context-aware AI applications.
Features

Local model execution using Ollama
Support for tinyllama and llama3.2 models
Custom data ingestion for RAG
Example scripts for document retrieval and response generation
Lightweight and easy-to-configure setup

Prerequisites

Python 3.8 or higher
Ollama installed (see Ollama Installation Guide)
A compatible system (Linux, macOS, or Windows) with sufficient storage for model files
Optional: CUDA-compatible GPU for faster inference with llama3.2

Installation

Clone the Repository:
git clone https://github.com/ankurpython/RAG_with_ollama_model.git
cd RAG_with_ollama_model


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Python Dependencies:Install required packages listed in requirements.txt (if provided) or the following core dependencies:
pip install langchain langchain-community sentence-transformers faiss-cpu


Install Ollama:Follow the instructions on the Ollama website to install Ollama for your operating system.

Pull Ollama Models:Download the tinyllama and llama3.2 models:
ollama pull tinyllama
ollama pull llama3.2


Verify Installation:Ensure models are installed:
ollama list



Usage

Prepare Custom Data:

Place your custom dataset (e.g., text files, PDFs) in the data/ directory.
Update the data loading script (e.g., load_data.py) to point to your dataset.


Run the RAG Pipeline:Execute the main script to process your data and generate responses:
python main.py


Ensure main.py (or equivalent) is configured with your model choice (tinyllama or llama3.2) and data path.


Example Query:
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()
# Load documents into FAISS
db = FAISS.from_documents(documents=chunks, embedding=embeddings)
# Query the model
llm = Ollama(model="tinyllama")
query = "What is the main topic of the dataset?"
result = db.similarity_search(query)
print(result)


Output:The script will retrieve relevant documents and generate a response based on the query and the selected model.



