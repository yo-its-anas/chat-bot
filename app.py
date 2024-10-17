# streamlit_chatbot.py

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set page configuration
st.set_page_config(
    page_title="Generative AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Hide Streamlit style
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"  # You can change this to "base" or "small" if you face resource issues
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Title and description
st.title("🤖 Generative AI Chatbot")
st.write("Ask me anything about **Generative AI**!")

# User input
with st.form(key='user_input_form'):
    user_input = st.text_input("You:", key="input")
    submit_button = st.form_submit_button(label='Send')

# When the user submits a question
if submit_button and user_input:
    # Append the user input to the conversation history
    st.session_state.history.append({"user": user_input})

    # Prepare the input for the model
    input_text = f"Answer this question about Generative AI: {user_input}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the response
    output_ids = model.generate(input_ids, max_length=200, num_beams=5, early_stopping=True)
    bot_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Append the bot response to the conversation history
    st.session_state.history.append({"bot": bot_response})

# Display conversation history
for message in st.session_state.history:
    if "user" in message:
        st.markdown(f"**You:** {message['user']}")
    else:
        st.markdown(f"**Chatbot:** {message['bot']}")
