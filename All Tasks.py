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
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.translate.bleu_score import sentence_bleu
import json
import pandas as pd
import numpy as np
import spacy

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("pinecone_api_key")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load SpaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Task 2: Design a Prompt
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

def get_openai_response(message, history):
    relevant_chunk = get_relevant_chunk(message)
    relevant_chunk_text = " ".join([chunk['metadata']['text'] for chunk in relevant_chunk['matches']])

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the following context only and answer the user query in a polite manner: {relevant_chunk_text}"},
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

# Task 3: Evaluation Pipeline
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

def calculate_faithfulness(answer, context):
    answer_claims = answer.split(".")
    context_claims = context.split(".")
    
    matched_claims = [claim for claim in answer_claims if any(claim.strip() in ctx for ctx in context_claims)]
    return len(matched_claims) / len(answer_claims) if answer_claims else 0

def calculate_answer_relevance(question, generated_answer):
    vectorizer = CountVectorizer().fit_transform([question, generated_answer])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def calculate_context_precision(context, ground_truth):
    context_chunks = context.split(".")
    gt_chunks = ground_truth.split(".")

    precision_scores = []
    for i, chunk in enumerate(context_chunks):
        if any(gt_chunk.strip() in chunk for gt_chunk in gt_chunks):
            precision_scores.append(1 / (i + 1))
        else:
            precision_scores.append(0)
    
    return np.mean(precision_scores)

def calculate_context_relevancy(context, ground_truth):
    context_sentences = context.split(".")
    relevant_sentences = [sentence for sentence in context_sentences if any(gt in sentence for gt in ground_truth.split("."))]
    return len(relevant_sentences) / len(context_sentences) if context_sentences else 0

def calculate_context_recall(context, ground_truth):
    gt_sentences = ground_truth.split(".")
    recalled_sentences = [sentence for sentence in gt_sentences if any(ctx in sentence for ctx in context.split("."))]
    return len(recalled_sentences) / len(gt_sentences) if gt_sentences else 0

def calculate_answer_semantic_similarity(answer, ground_truth):
    vectorizer = CountVectorizer().fit_transform([answer, ground_truth])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def calculate_answer_correctness(answer, ground_truth):
    factual_correctness = calculate_faithfulness(answer, ground_truth)
    semantic_similarity = calculate_answer_semantic_similarity(answer, ground_truth)
    return (factual_correctness + semantic_similarity) / 2

def evaluate_all_metrics(question, generated_answer, context, ground_truth):
    metrics = {}
    metrics['Faithfulness'] = calculate_faithfulness(generated_answer, context)
    metrics['Answer Relevance'] = calculate_answer_relevance(question, generated_answer)
    metrics['Context Precision'] = calculate_context_precision(context, ground_truth)
    metrics['Context Relevancy'] = calculate_context_relevancy(context, ground_truth)
    metrics['Context Recall'] = calculate_context_recall(context, ground_truth)
    metrics['Answer Semantic Similarity'] = calculate_answer_semantic_similarity(generated_answer, ground_truth)
    metrics['Answer Correctness'] = calculate_answer_correctness(generated_answer, ground_truth)
    
    return metrics

def save_evaluation_results(results, filename="evaluation_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def save_evaluation_results_csv(results, filename="evaluation_results.csv"):
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)

# Task 4: Integrate to Gradio UI
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
    gr.Markdown("# AI Chatbot powered by OpenAI")

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
    # Example evaluation
    question = "Where was Albert Einstein born?"
    generated_answer = "Einstein was born in Germany in 1879."
    context = "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist."
    ground_truth = "Einstein was born in Germany on March 14, 1879."
    
    metrics = evaluate_all_metrics(question, generated_answer, context, ground_truth)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Save evaluation results
    save_evaluation_results(metrics)
    save_evaluation_results_csv(metrics)
    
    # Launch the Gradio app
    demo.launch()
