import streamlit as st
import pickle
import random

# Load the trained model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define chatbot responses
positive_responses = [
    "😊 I'm glad you're feeling good!",
    "🌟 That's great to hear! Keep up the positivity!",
    "😃 Sounds like you're having a great day!"
]

negative_responses = [
    "💙 I'm sorry you're feeling this way. Remember, you're not alone.",
    "🤗 It's okay to feel down sometimes. Want to talk about it?",
    "🫂 I'm here to listen. What's on your mind?"
]

# Function to generate chatbot response & sentiment label
def get_chatbot_response(user_input):
    user_input_tfidf = vectorizer.transform([user_input])
    sentiment = model.predict(user_input_tfidf)[0]
    
    if sentiment == 1:
        return random.choice(positive_responses), "🙂 Positive Sentiment"
    else:
        return random.choice(negative_responses), "☹️ Negative Sentiment"

# Set Streamlit page config
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage { font-size: 16px !important; }
    .stChatMessage.user { background-color: #d1ecf1; border-radius: 10px; padding: 10px; }
    .stChatMessage.assistant { background-color: #f8d7da; border-radius: 10px; padding: 10px; }
    .sentiment-label { font-size: 14px; color: #555; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with extra features
st.sidebar.title("🔹 About")
st.sidebar.info("This is a Mental Health Chatbot that detects sentiment and provides supportive responses.")
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []

# Main UI
st.title("🧠 Mental Health Chatbot")
st.write("💬 Talk to me! Tell me how you feel.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sentiment" in message:
            st.markdown(f'<p class="sentiment-label">{message["sentiment"]}</p>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Ensure user message is displayed before generating a response
    user_message = {"role": "user", "content": "👤 " + user_input}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(user_message["content"])

    # Get chatbot response and sentiment label
    response, sentiment_label = get_chatbot_response(user_input)

    # Ensure chatbot response and detected sentiment are displayed
    chatbot_message = {"role": "assistant", "content": "🤖 " + response, "sentiment": sentiment_label}
    st.session_state.messages.append(chatbot_message)

    with st.chat_message("assistant"):
        st.write(chatbot_message["content"])
        st.markdown(f'<p class="sentiment-label">{chatbot_message["sentiment"]}</p>', unsafe_allow_html=True)
