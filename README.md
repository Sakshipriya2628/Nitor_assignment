# Nitor_project
# RAG-Based AI Application

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based AI application. The application extracts text from PDF documents, creates embeddings, stores them in a vector database, and uses this information to answer user queries. The project is divided into four main tasks:

1. Document Extraction
2. Prompt Design
3. Response Evaluation
4. Gradio UI Integration

## Task 1: Document Extraction

### Approach

The document extraction process is implemented in the `text_extraction_and_embedding.py` file. Here's a breakdown of the approach:

1. **PDF to Image Conversion:** The PDF is converted to images using the `pdf2image` library.
2. **Text Extraction:** Tesseract OCR is used to extract text from the images.
3. **Text Chunking:** The extracted text is divided into overlapping chunks for better context preservation.
4. **Embedding Creation:** OpenAI's API is used to create embeddings for each text chunk.
5. **Vector Database Storage:** The embeddings are stored in a Pinecone vector database for efficient retrieval.

## Task 2: Prompt Design

### Approach

The prompt design is implemented in the All_Tasks.py file. The approach involves:

1. Relevant Chunk Retrieval: Given a user query, retrieve relevant text chunks from the Pinecone database.
2. Context-Based Prompt: Create a prompt that includes the relevant context and the user's question.
3. OpenAI API Integration: Use OpenAI's API to generate a response based on the prompt.


## Task 3: Evaluating Responses

### Approach

The evaluation pipeline is designed to assess the quality of the generated responses. It includes several metrics:

1.Faithfulness
2.Answer Relevance
3.Context Precision
4.Context Relevancy
5.Context Recall
6.Answer Semantic Similarity
7.Answer Correctness


## Task 4: Gradio UI Integration

### Approach

The Gradio UI integrates the document extraction, prompt design, and response generation into a user-friendly interface. It includes:

1. A chat interface for user interactions
2. Integration with the OpenAI response generation
3. Custom CSS for improved appearance

## Conclusion

This project demonstrates a comprehensive approach to building a RAG-based AI application. It covers the entire pipeline from document extraction to user interaction, including evaluation metrics for assessing the quality of generated responses. The modular design allows for easy modifications and improvements in each component of the system.

