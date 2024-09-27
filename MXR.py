import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

#Loading and preprocessing dataset
df = pd.read_csv('C:/Users/jorda/OneDrive/Desktop/MotionX robotics/sample_data.csv')  

#Map labels to integers if not  done
df['label'] = df['label'].map({"positive": 1, "negative": 0})

#Convert it to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

#Loading the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("aychang/roberta-base-imdb")
model = RobertaForSequenceClassification.from_pretrained("aychang/roberta-base-imdb", num_labels=2)

#Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, return_tensors='pt')

#Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

#Split into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

#Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

#Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

#Train the model
trainer.train()

#Evaluate the model
results = trainer.evaluate()
print(results)

#Saveing the files (fine-tuned model and tokenizer)
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
