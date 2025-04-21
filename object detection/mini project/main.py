from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

app = Flask(__name__)

# Create a mock dataset (same as your original code)
data = {
    'symptom_fever': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    'symptom_chills': [1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    'symptom_headache': [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    'symptom_muscle_pain': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'malaria': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 for infected, 0 for not infected
}

df = pd.DataFrame(data)
X = df[['symptom_fever', 'symptom_chills', 'symptom_headache', 'symptom_muscle_pain']]
y = df['malaria']

# Build the model
model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (In a real application, you would probably load a pre-trained model instead)
model.fit(X, y, epochs=5, batch_size=5, verbose=0)

@app.route('/predict', methods=['POST'])
def predict_malaria():
    data = request.get_json()  # Get JSON data from the request
    symptoms = np.array([[data['symptom_fever'], data['symptom_chills'],
                           data['symptom_headache'], data['symptom_muscle_pain']]])
    
    prediction = model.predict(symptoms)
    predicted_class = (prediction[0][0] > 0.5).astype(int)  # Threshold of 0.5 for classification
    
    result = "Malaria Detected" if predicted_class == 1 else "No Malaria Detected"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
