from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import requests
from groq import Groq
import markdown

app = Flask(__name__)

# Load the trained model for disease prediction
model = joblib.load('model.pkl')

# OpenFDA API key for fetching medicine information
api_key = "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"

# Function to get medicine info using OpenFDA API
def get_medicine_info(disease):
    url = f'https://api.fda.gov/drug/label.json?search={disease}&limit=5'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        medicines = []
        if 'results' in data and len(data['results']) > 0:
            for result in data['results']:
                if 'openfda' in result and 'generic_name' in result['openfda']:
                    medicines.extend(result['openfda']['generic_name'])
                if len(medicines) >= 3:
                    break
            return medicines[:3] if medicines else ["No medicine found for this disease"]
        else:
            return ["No medicine found for this disease"]
    
    except Exception as e:
        print(f"Error with OpenFDA API: {e}")
        return ["Error fetching medicine information"]

# Route for the home page
@app.route('/')
def home():
    return render_template('Medicine_Finder.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = [
            'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
            'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
            'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
            'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
            'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
            'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
            'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
            'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 
            'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelled_lymph_nodes', 
            'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 
            'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 
            'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 
            'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 
            'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 
            'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 
            'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 
            'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 
            'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
            'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
            'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 
            'red_spots_over_body', 'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 
            'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 
            'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 
            'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 
            'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 
            'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
            'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'
        ]

        # Create input data for the model
        input_data = [1 if symptom in request.form.getlist('symptoms') else 0 for symptom in symptoms]
        input_data.append(0)  # Default for fluid_overload.1 if not in symptoms
        input_df = pd.DataFrame([input_data], columns=symptoms + ['fluid_overload.1'])

        # Ensure the columns match the training model's expectations
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict disease using the trained model
        prediction = model.predict(input_df)[0]  # Predict the disease

        # Get medicine information based on the predicted disease
        medicine_info = get_medicine_info(prediction)

        # Prepare data for the next step: get diet plan, precautions, etc.
        data = {
            "age": request.form.get("age"),
            "gender": request.form.get("gender"),
            "symptoms": ", ".join([symptom for symptom, selected in zip(symptoms, input_data) if selected]),
            "disease": prediction,
            "location": request.form.get("location")
        }

        # Call the get_plan API to get diet plan, precautions, and daily routine
        plan_response = requests.post(f"http://127.0.0.1:5000/get_plan", json=data)
        plan_data = plan_response.json()

        # Render the result page with all data: disease, medicine info, and plan details
        return render_template('result.html', 
                               diagnosis=prediction, 
                               medicines=medicine_info,
                               plan=plan_data['result'],
                               symptoms=data['symptoms'])

# Route for the index page (diet plan)
@app.route('/index.html')
def index():
    return render_template('index.html')

# Route for generating the diet plan, precautions, and daily routine based on user input
@app.route('/get_plan', methods=['POST'])
def get_plan():
    data = request.json
    age = data['age']
    gender = data['gender']
    symptoms = data['symptoms']
    disease = data['disease']
    location = data['location']
    
    client = Groq(api_key=api_key)
    prompt = (
        f"Provide a general diet plan, precautions, and daily routine for a "
        f"{age}-year-old {gender} with {disease} and symptoms of {symptoms}. "
        f"Provide the Medical Store Name, in new line its address, and in new line its phone number which location is this {location}. "
        f"Give Only 3 Medical Stores in that Region. "
        f"Do not provide any introduction. Use above response in proper heading and subheading and use Border without forgetting."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )

        response_text = chat_completion.choices[0].message.content
        markdown_content = response_text.strip()
        html_content = markdown.markdown(markdown_content)

        return jsonify({"result": html_content})

    except Exception as e:
        print(f"Error while fetching diet plan: {str(e)}")
        return jsonify({"error": str(e)})

# Route for the medicine finder page
@app.route('/medicine_finder')
def medicine_finder():
    return render_template('Medicine_Finder.html')

if __name__ == '__main__':
    app.run(debug=True)
