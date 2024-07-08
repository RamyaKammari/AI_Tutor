import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import warnings
import requests
import os
from huggingface_hub import InferenceClient  # Added import
import time
from huggingface_hub.utils import HfHubHTTPError
warnings.filterwarnings("ignore")

load_dotenv()
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
MIXTRAL_API_KEY = os.getenv("MIXTRAL_API_KEY")  # Added environment variable for Mixtral API key

def query_huggingface_api(prompt):
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    response = requests.post(HUGGING_FACE_API_URL, headers=headers, json={"inputs": prompt})
    print(response.json())
    return response.json()[0]['generated_text']

def query_mixtral_api(prompt):
    client = InferenceClient(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=MIXTRAL_API_KEY,
    )
    response_text = ""
    retries = 3
    for attempt in range(retries):
        try:
            for message in client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                stream=True,
            ):
                response_text += message.choices[0].delta.content
            return response_text
        except Exception as e:
            print(e)
            print(e.response.status_code)
            if e.response.status_code == 429:
                print(f"Rate limit reached. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                raise e
    raise Exception("Failed to get a response from Mixtral API after several retries.")

def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query the database and generate a response using the specified model.")
    parser.add_argument("--query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, choices=["openai", "llama", "mixtral"], default="openai", help="The model to use for generating responses. Choices are 'openai', 'llama', or 'mixtral'. Default is 'openai'.")
    args = parser.parse_args()
    query_text = args.query_text
    model_choice = args.model

    # Example usage:
    # python query_data\ copy.py "What is the capital of France?" --model llama
    # python query_data\ copy.py "What are the laws of chemical combinations?" --model openai

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    if model_choice == "openai":
        model = ChatOpenAI()
        response_text = model.predict(prompt)
    elif model_choice == "llama":
        response_text = query_huggingface_api(prompt)
    elif model_choice == "mixtral":
        response_text = query_mixtral_api(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()