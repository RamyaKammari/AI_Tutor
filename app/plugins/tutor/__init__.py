import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
import requests
import time
from huggingface_hub import InferenceClient
from app.commands import Command
from langchain_groq import ChatGroq  # Import Groq library

class TutorAgent(Command):
    def __init__(self):
        super().__init__()
        self.name = "tutor"
        self.description = "Interact with a Tutor AI to explore educational topics."
        self.history = []
        load_dotenv()
        self.API_KEY = os.getenv('OPEN_AI_KEY')
        self.HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
        self.llm = ChatOpenAI(openai_api_key=self.API_KEY, model="gpt-4-0125-preview")  # Initialize once and reuse
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(persist_directory="chroma", embedding_function=self.embedding_function)
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.groq_llm = ChatGroq(api_key=self.GROQ_API_KEY, model="llama3-70b-8192")  # Initialize Groq model

    def calculate_tokens(self, text):
        return len(text) + text.count(' ')

    def query_mixtral_api(self, prompt):
        client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=self.HUGGING_FACE_API_KEY)
        response_text = ""
        retries = 3
        for attempt in range(retries):
            try:
                for message in client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=500, stream=True):
                    response_text += message.choices[0].delta.content
                return response_text
            except Exception as e:
                if e.response.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        raise Exception("Failed to get a response from Mixtral API after several retries.")

    def interact_with_ai(self, user_input, character_name, model_choice="mixtral"):
        print("----\n HISTORY \n----")
        print(self.history)
        print("----\n END HISTORY \n----")
        query_text = str(self.history) 
        results = self.db.similarity_search_with_relevance_scores(query_text, k=5)

        print("----\n RESULTS \n----")
        print(results)
        print("----\n END RESULTS \n----")


        sources = [doc.metadata.get("source", None) for doc, _score in results]
        print("----\n SOURCES \n----")
        print(sources)
        print("----\n END SOURCES \n----")

        if len(results) == 0 or results[0][1] < 0.7:
            return "Unable to find matching results.", 0, []

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_text = f"""
        You are a highly knowledgeable Tutor AI with expertise in chemistry especially for IIT JEE examination. Your role is to provide accurate, detailed, and context-based answers to the user's questions. Refer only to the following context to ensure your response is relevant and precise.

        Context:
        {context_text}

        previous conversation: 
        {self.history[:-1]}

        Question:
        {self.history[-1][1]}

        Your answer should be based on the above context and structured in a clear, concise, and informative manner.
        """

        print("-----\n PROMPT TEXT \n-----")
        print(prompt_text)
        print("-----\n END PROMPT TEXT \n-----")

        output_parser = StrOutputParser()

        if model_choice == "openai":
            print("Using Model: OpenAI")
            chain = ChatPromptTemplate.from_messages(self.history + [("system", prompt_text)]) | self.llm | output_parser
            response = chain.invoke({"input": user_input})
        elif model_choice == "mixtral":
            print("Using Model: Mixtral")
            response = self.query_mixtral_api(prompt_text)
        elif model_choice == "groq":
            print("Using Model: Groq")
            prompt = ChatPromptTemplate.from_messages([("human", prompt_text)])
            chain = prompt | self.groq_llm
            start_time = time.time()
            response = chain.invoke({"input": user_input}).content
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken by the LLM call: {time_taken:.2f} seconds")
        else:
            print("Invalid model choice. Defaulting to OpenAI.")
            chain = ChatPromptTemplate.from_messages(self.history + [("system", prompt_text)]) | self.llm | output_parser
            response = chain.invoke({"input": user_input})
        
        print("-----\n RESPONSE \n-----")
        print(response)
        print("-----\n END RESPONSE \n-----")
        tokens_used = self.calculate_tokens(prompt_text + user_input + response)
        logging.info(f"API call made. Tokens used: {tokens_used}")
        return response, tokens_used, list(set(sources))

    def execute(self, *args, **kwargs):
        character_name = kwargs.get("character_name", "Tutor")
        model_choice = kwargs.get("model_choice", "mixtral")
        print(f"Welcome to the JEE Chemistry AI Tutor! your personal buddy to help and explain your doubts in chemistry. Type 'done' to exit anytime.")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "done":
                print("Thank you for using the JEE Chemistry AI Tutor. Goodbye!")
                break

            self.history.append(("user", user_input))

            try:
                response, tokens_used, sources = self.interact_with_ai(user_input, character_name, model_choice)
                print(f"Tutor: {response}")
                # print(f"(This interaction used {tokens_used} tokens.)")
                self.history.append(("system", response))
            except Exception as e:
                print("Sorry, there was an error processing your request. Please try again.")
                logging.error(f"Error during interaction: {e}")

if __name__ == "__main__":
    tutor_agent = TutorAgent()
    tutor_agent.execute()