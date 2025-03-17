from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (replace 'final_model.h5' with your model file name)
model = load_model(r'C:\happy\college\final-ml\Final_project\flask\ckd_ensemble_model.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.form.to_dict()
        
        # Extract known features
        input_features = np.array([[
            float(data['age']),
            float(data['bp']),
            float(data['sg']),
            float(data['al']),
            float(data['su'])
        ]])
        
        # Add placeholders for the missing features (e.g., zeros)
        missing_features = np.zeros((1, 5))  # Adjust based on how many are missing
        input_features = np.concatenate([input_features, missing_features], axis=1)

        # Reshape input for the model
        input_reshaped = np.reshape(input_features, (input_features.shape[0], 1, input_features.shape[1]))

        # Make prediction
        prediction_prob = model.predict(input_reshaped)
        prediction_prob_value = prediction_prob[0][0]
        
        # Classify based on the probability
        if prediction_prob_value < 0.10:
            result_message = "No Chronic Kidney Disease"
        else:
            result_message = "Chronic Kidney Disease"

        # Prepare response
        result = {
            'prediction': result_message,
            'probability': f"{prediction_prob_value:.2f}"
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})


    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)