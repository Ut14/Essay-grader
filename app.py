from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, AutoTokenizer

app=Flask(__name__)
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

@app.route('/')
def home():
    """Render the home page with a form for input."""
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    """Handle the POST request and return the prediction."""
    try:
        data = request.json  # Get the JSON data from the POST request
        question = data.get('question')  # Extract the question
        essay_text = data.get('essay')  # Extract the essay text

        if question and essay_text:
            predicted_score = predict_essay_score(essay_text)
            predicted_score = float(predicted_score)

            response = {
                'status': 'success',
                'data': {
                    'question': question,
                    'essay': essay_text,
                    'predicted_score': predicted_score,
                    'message': 'Prediction completed successfully.'
                },
                'metadata': {
                    'model': 'BERT-based Sequence Classifier',
                    'version': '1.0',
                    'confidence': 'N/A'  # Add confidence score if applicable
                }
            }
            return jsonify(response)
        else:
            return jsonify({"error": "Both question and essay text are required"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)