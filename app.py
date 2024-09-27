import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_model")

st.title("Sentiment Insight Chatbot")

# User input
user_input = st.text_input("Enter your message:")

if st.button("Submit"):
    inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"The sentiment of your message is: **{sentiment}**")
