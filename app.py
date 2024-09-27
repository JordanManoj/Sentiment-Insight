import os  # Import the os module
import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"  # Update this path if necessary
if not os.path.exists(model_path):
    st.error(f"Model path {model_path} does not exist!")
else:
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Streamlit interface
st.title("Sentiment Analysis Chatbot")
user_input = st.text_input("Enter a text to analyze sentiment:")
if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Sentiment: {sentiment}")
