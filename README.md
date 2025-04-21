# Overview

The AI Technical assessment project to demonstrate various document processing functionalities using AI models. The project includes capabilities for document reading, translation, summarization, and question-answering based on the content of publications. It leverages advanced NLP techniques and models to provide efficient and accurate results.

# Features

- **Document Reading**: Supports reading and extracting text from various document formats including DOCX, PDF, CSV, XLSX, XLS, and XLSM.
- **Question & Answer**: Uses a vector database to store document chunks and provides answers to user queries based on the stored information.
- **Document Translation**: Translates documents into English or Arabic.
- **Document Summarization**: Offers extractive, abstractive, and hybrid summarization techniques to generate concise summaries of documents.
- **Performance Metrics**: Tracks and calculates performance metrics for NLP operations to ensure efficiency.

# Installation

1. Clone the repository:
   ```bash
     git clone https://github.com/Abdallah-Elghandour/-Dr.-X-Assignment.git
  
2. Create a virtual environment and install dependencies:
  ```bash
    python -m venv venv
    source venv/bin/activate  # or .\venv\Scripts\activate on Windows
    pip install -r requirements.txt
  ```
3. Run the Main Program:
  ```bash
    python src/main.py
  ```


# Methodology

The project implements a comprehensive document processing pipeline with the following components:

## 1. Document Processing

- Documents are read using format-specific parsers:
  - **PDFs**
  - **DOCX**
  - **Tabular data**
- DOCX files are converted to PDFs to handle pages numbers.
- Text is extracted from tables.
- Documents are chunked into manageable segments for processing.

## 2. Vector Database

- Document chunks are embedded using the **Nomic AI embedding model** `nomic-embed-text-v2-moe`.
- Embeddings are stored in a **FAISS** vector database for efficient similarity search.
- **Retrieval-Augmented Generation (RAG)** is used to provide context-aware answers.

## 3. Question & Answer

- User queries are embedded using the same embedding model.
- Similar documents are retrieved from the vector database.
- Relevant documents are used to generate answers using the **Llama 3.1 8b**.

## 4. Translation

- Documents are translated as its original structure and format for (PDF and DOCX).
- Translator is using `facebook/nllb-200-distilled-600M model`.
- Supports high-quality translation for English and Arabic.


## 5. Summarization

- Documents are chunked to manage token limits.
- Each chunk is summarized using the **Llama 3.1 8b** model.
- **ROUGE metrics** are used to evaluate summary quality when reference summaries are available.

# Models Used

## Large Language Model (LLM)

- **Model**: Meta's Llama 3.1 8B Instruct  
- **Source**: Hugging Face `meta-llama/Llama-3.1-8B-Instruct`  
- **Quantization**: 4-bit quantization using `bitsandbytes` to reduce memory requirements  
- **Implementation**: Loaded using Hugging Face Transformers with inference optimizations

## Translation Model

- **Model**: NLLB-200 Distilled 
- **Source**: Hugging Face `facebook/nllb-200-distilled-600M`  
- **Purpose**: Provides multilingual translation, including English and Arabic
- **Implementation**: Loaded using Hugging Face Transformers



## Embedding Model

- **Model**: Nomic AI embedding model
- **Source**: Hugging Face `nomic-ai/nomic-embed-text-v2-moe` 
- **Purpose**: Converts text chunks into high-dimensional vector representations  
- **Implementation**: Loaded using Hugging Face Sentence Transformers with inference optimizations

# Significant Discoveries

## 1. Performance Optimization

- Chunking large documents significantly improves processing efficiency.
- 4-bit quantization enables running 8B parameter models on Laptop RTX 4070 8GB.
- Performance metrics show an average processing speed of **X tokens per second**.

## 2. Translation Quality

- `facebook/nllb-200-distilled-600M` provides reliable translations for English and Arabic.

## 3. Summarization Effectiveness

- Optimal chunk size for summarization is approximately **4000 tokens**.

# Usage

The application provides a command-line interface with the following options:

1. **Question & Answer**  
   Ask questions about the content of loaded documents.

2. **Document Translation**  
   Translate documents between English and Arabic.

3. **Document Summarization**  
   Generate summaries using different techniques.
