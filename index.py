import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('./suicide_classifier')

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 80))

@app.route('/predict', methods=['GET'])
def predict():
  input_text = request.args.get('text', '')
  if not input_text:
    return jsonify({'error': 'Please provide input text as a parameter.'}), 400
  
  prediction = model.predict([input_text])
  
  return jsonify(prediction.tolist()[0][0] * 100), 200

if __name__ == '__main__':
  app.run(host=HOST, port=PORT)
