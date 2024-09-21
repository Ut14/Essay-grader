import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, AutoTokenizer
model = TFBertForSequenceClassification.from_pretrained("best_model")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

def predict_essay_score(essay_text):
    # Tokenize the essay input
    inputs = tokenizer([essay_text], padding='max_length', truncation=True, return_tensors="tf")

    # Run the model prediction
    predictions = model(inputs)

    # Extract the predicted score
    predicted_score = predictions.logits.numpy()[0][0]
    return predicted_score

# Ask the user to input the question and essay
question = input("Please enter the question: ")
essay = input("Please enter your essay: ")

# Use the trained model to predict the score
predicted_score = predict_essay_score(essay)

# Output the result
print("\nPredicted Score for the given essay: ", predicted_score)