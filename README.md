
# Sentiment-Insight

Sentiment-Insight is a sentiment analysis chatbot that leverages a fine-tuned RoBERTa model for binary classification of sentiments (positive or negative). This project allows users to input text and instantly receive feedback on whether the sentiment is positive or negative, powered by a pretrained model from Hugging Face.



## Project Overview
Sentiment-Insight is built using the Hugging Face Transformers library, leveraging the aychang/roberta-base-imdb model for sentiment analysis. The project involves the following steps:

- Fine-tuning the Model: The RoBERTa model is fine-tuned on a custom sentiment dataset.

- Tokenization: Text input is tokenized using a pre-trained tokenizer from the model.

- Prediction: The fine-tuned model predicts whether the sentiment of the input text is positive or negative.

- Deployment: The app is deployed using Streamlit to provide a web-based interface.
## Features
- Interactive Chatbot: Users can interact with the chatbot via a simple web interface, entering text to analyze its sentiment.

- Pre-trained Model: Uses the fine-tuned version of Roberta-base trained on IMDB sentiment data for efficient and accurate predictions.

- On-the-fly Analysis: Analyze any text and get sentiment predictions instantly.
## Instruction
### Step 1:

Save the data set and the python files (MXR.py , app.py) onto your system

### Step 2:

Provide the file path of the dataset in the code(MXR.py)

### Step 3:

After runing the code (MXR.py), the requied finetuned model will be save to the system

### Step 4:

Then run the "app.py" file and enter "streamlit run app.py" in the terminal

### Step 5:

Once the app is running, open your web browser and navigate to "http://localhost:8501" . You can interact with the chatbot and analyze sentiment by entering text in the input field.
## Fine-Tuning the Model

The app.py script uses the pre-trained aychang/roberta-base-imdb model for inference. To fine-tune it further:

1. Replace or update 'sample_data.csv' with your custom dataset.
1. Modify the training parameters in the code as needed.
1. Run the training process by adding code for model training using Hugging Face’s Trainer API.
## Contributing

Feel free to fork the repository and submit pull requests if you have suggestions for improvements!