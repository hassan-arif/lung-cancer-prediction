import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

app = Flask(__name__)

# Load the trained machine learning model
model = load_model('my_model.h5')

# Load the scaler object used during training
scaler = StandardScaler()
scaler_file = 'scaler.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html', favicon_url=url_for('static', filename='images/favicon.png'))

# Define the form page route
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Print out the form data for debugging
        print(request.form)

        # Extract the form data
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        if gender == 'Male': gender = 1
        else: gender = 2
        air_pollution = int(request.form.get('air_pollution'))
        alcohol_use = int(request.form.get('alcohol_use'))
        dust_allergy = int(request.form.get('dust_allergy'))
        occupational_hazards = int(request.form.get('occupational_hazards'))
        genetic_risk = int(request.form.get('genetic_risk'))
        chronic_lung_disease = int(request.form.get('chronic_lung_disease'))
        balanced_diet = int(request.form.get('balanced_diet'))
        obesity = int(request.form.get('obesity'))
        smoking = int(request.form.get('smoking'))
        passive_smoker = int(request.form.get('passive_smoker'))
        chest_pain = int(request.form.get('chest_pain'))
        coughing_blood = int(request.form.get('coughing_blood'))
        fatigue = int(request.form.get('fatigue'))
        weight_loss = int(request.form.get('weight_loss'))
        shortness_of_breath = int(request.form.get('shortness_of_breath'))
        wheezing = int(request.form.get('wheezing'))
        swallowing_difficulty = int(request.form.get('swallowing_difficulty'))
        clubbing = int(request.form.get('clubbing'))
        frequent_cold = int(request.form.get('frequent_cold'))
        dry_cough = int(request.form.get('dry_cough'))
        snoring = int(request.form.get('snoring'))

        data = np.zeros((23))
        data[0] = age
        data[1] = gender
        data[2] = air_pollution
        data[3] = alcohol_use
        data[4] = dust_allergy
        data[5] = occupational_hazards
        data[6] = genetic_risk
        data[7] = chronic_lung_disease
        data[8] = balanced_diet
        data[9] = obesity
        data[10] = smoking
        data[11] = passive_smoker
        data[12] = chest_pain
        data[13] = coughing_blood
        data[14] = fatigue
        data[15] = weight_loss
        data[16] = shortness_of_breath
        data[17] = wheezing
        data[18] = swallowing_difficulty
        data[19] = clubbing
        data[20] = frequent_cold
        data[21] = dry_cough
        data[22] = snoring

        # Convert the list to a numpy array with the desired shape
        new_data = np.array([data])

        # Standardize the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)

        # Make predictions
        predictions = model.predict(new_data_scaled)

        # Convert the predictions to class labels
        predicted_classes = np.argmax(predictions, axis=1)


        # Determine the predicted outcome based on the prediction
        if predicted_classes[0] == 0:
            outcome = 'Low'
        elif predicted_classes[0] == 1:
            outcome = 'Medium'
        else:
            outcome = 'High'

        # Render the results template with the predicted outcome
        return render_template('results.html', outcome=outcome)
    else:
        # If the request method is not POST, redirect to the home page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
