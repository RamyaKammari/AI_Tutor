import streamlit as st
from app.plugins.tutor import TutorAgent
import time

# Initialize the TutorAgent
if 'tutor_agent' not in st.session_state:
    st.session_state.tutor_agent = TutorAgent()

if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "mixtral"

tutor_agent = st.session_state.tutor_agent

# Streamlit app layout
st.title("JEE Chemistry AI Tutor", anchor="center")
st.write("Welcome to the JEE Chemistry AI Tutor! Your personal buddy to help and explain your doubts in chemistry.")

model_choice = st.selectbox("Choose model", ["mixtral", "openai", "groq"])
st.session_state.model_choice = model_choice

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if("caption" in message):
            st.caption(message["caption"])


if prompt := st.chat_input("Welcome and ask a question to the AI tutor"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    tutor_agent.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar='👨🏻‍🏫'):
        response_placeholder = st.empty()
        caption_placeholder = st.empty()
        start_time = time.time()
        with st.spinner('Thinking...'):
            response, tokens_used, sources = tutor_agent.interact_with_ai(prompt, "Tutor", st.session_state.model_choice)
        elapsed_time = round(time.time() - start_time, 2)
        caption = f"**Sources:** {sources} \n**Time taken:** {elapsed_time} seconds\t\t\t**Tokens used:** {tokens_used}\t\t\t**Model:** {st.session_state.model_choice.capitalize()}"
        response_placeholder.markdown(response)
        caption_placeholder.caption(caption)  # Ensure caption is set only once
        st.session_state.messages.append({"role": "assistant", "content": response, "caption": caption})
        tutor_agent.history.append(("system", response))