import pickle
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler with error handling
try:
    with open('model/voting_model.pkl', 'rb') as model_file:
        voting_model = pickle.load(model_file)
    with open('model/preprocessing_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    voting_model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if voting_model is None or scaler is None:
            return jsonify({'success': False, 'error': 'Model not loaded properly'})
            
        # Extract input features from form data
        input_features = [float(request.form.get(feature)) for feature in request.form.keys()]
        input_array = np.array([input_features])
        
        # Separate Transaction_Date (year) and other features
        transaction_date = input_array[:, 0].reshape(-1, 1)
        features_to_scale = input_array[:, 1:]
        
        # Scale the input features
        scaled_features = scaler.transform(features_to_scale)
        
        # Combine Transaction_Date with the scaled features
        final_input = np.hstack([transaction_date, scaled_features])
        
        prediction = voting_model.predict(final_input)
        price_per_sq_meter_ntd = round(prediction[0], 2)
        
        return jsonify({'success': True, 'price': price_per_sq_meter_ntd})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)