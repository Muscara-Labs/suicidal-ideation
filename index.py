import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)
model = tf.keras.models.load_model('./suicide_classifier')

@app.route('/predict', methods=['GET'])
def predict():
  input_text = request.args.get('text', '')
  if not input_text:
    return jsonify({'error': 'Please provide input text as a parameter.'}), 400
  
  prediction = model.predict([input_text])
  
  return jsonify(prediction.tolist()[0][0] * 100), 200

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)
