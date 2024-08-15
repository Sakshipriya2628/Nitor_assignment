 
# Task 1: Document Extraction (use above PDF URL): Use different pdf parsing techniques to parse
# document text, images and table data from above documents. Use appropriate chunking strategy to
# chunk document and ingest data into Vector DB. (30 points)
# Task 2: Design a prompt: Design a prompt for a RAG based Generative AI application where user will
# get response based on the query. You must utilize the above Task 1 data and vector DB for generating
# responses. (30 points)
# Task 3: Evaluating Responses: Design evaluation pipeline for above task 2. Came up with some
# strategy which can be used to evaluate generated response from AI model. (20 points)
# Task 4: Integrate to Gradio UI: Try to integrate task 1 & task 2 with the Gradio UI. (20 points)
import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import textwrap
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
import json
import pandas as pd

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("pinecone_api_key")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=openai_api_key)

# Function to get relevant chunk from Pinecone vector database
def get_relevant_chunk(query):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("projecty")
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    relevant_chunk = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=2,
        include_values=True,
        include_metadata=True
    )

    return relevant_chunk

# Function to generate response from OpenAI model
def get_openai_response(message, history):
    relevant_chunk = get_relevant_chunk(message)
    relevant_chunk_text = " ".join([chunk['metadata']['text'] for chunk in relevant_chunk['matches']])

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the following context and answer the user in a polite way: {relevant_chunk_text}"},
        {"role": "user", "content": message}
    ]
    
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0
    )

    return response.choices[0].message.content.strip()

# Evaluation functions
def compute_bleu_score(reference, response):
    reference_tokens = [ref.split() for ref in reference]
    response_tokens = response.split()
    score = sentence_bleu(reference_tokens, response_tokens)
    return score

def compute_similarity_score(query, response):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, response])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity

def evaluate_responses(ground_truth, ai_responses):
    evaluation_results = []
    for query, correct_response in ground_truth.items():
        ai_response = ai_responses.get(query, "")
        bleu_score = compute_bleu_score([correct_response], ai_response)
        similarity_score = compute_similarity_score(correct_response, ai_response)
        evaluation_results.append({
            "query": query,
            "correct_response": correct_response,
            "ai_response": ai_response,
            "bleu_score": bleu_score,
            "similarity_score": similarity_score
        })
    return evaluation_results

def save_evaluation_results(results, filename="evaluation_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def save_evaluation_results_csv(results, filename="evaluation_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

# Sample ground truth and AI responses for testing (replace with actual data)
ground_truth = {
    "What is the capital of France?": "The capital of France is Paris.",
    "Who wrote 'To Kill a Mockingbird'?": "Harper Lee wrote 'To Kill a Mockingbird'."
}

# Function to get AI responses (mock responses for evaluation example)
def get_ai_responses():
    responses = {
        "What is the capital of France?": "Paris is the capital of France.",
        "Who wrote 'To Kill a Mockingbird'?": "The book 'To Kill a Mockingbird' was written by Harper Lee."
    }
    return responses

# Evaluate responses and save results
ai_responses = get_ai_responses()
results = evaluate_responses(ground_truth, ai_responses)
save_evaluation_results(results)
save_evaluation_results_csv(results)

print("Evaluation complete. Results saved.")

# Gradio UI integration
custom_css = """
.container {
    max-width: 800px !important;
    margin: auto;
    padding-top: 2rem;
}
.chat-window {
    height: 400px !important;
    overflow-y: auto;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 1rem;
    background-color: #f9f9f9;
}
.user-message {
    background-color: #DCF8C6;
    padding: 0.5rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    max-width: 70%;
    float: right;
    clear: both;
}
.bot-message {
    background-color: #FFFFFF;
    padding: 0.5rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    max-width: 70%;
    float: left;
    clear: both;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# AI Chatbot powered by OpenAI with PDF Support")

    chatbot = gr.Chatbot(height=400, container=True, bubble_full_width=False)
    msg = gr.Textbox(placeholder="Type your message here...", label="User Input")
    clear = gr.Button("Clear Chat")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = get_openai_response(history[-1][0], history[:-1])
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch()
