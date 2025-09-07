import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key


# ============================================================
# Database Setup
# ============================================================
def init_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  symptoms TEXT,
                  predicted_disease TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()


init_db()

# ============================================================
# Load Data & Model
# ============================================================
try:
    # Load all datasets and the model
    sym_des = pd.read_csv("symtoms_df.csv")
    precautions = pd.read_csv("precautions_df.csv")
    workout = pd.read_csv("workout_df.csv")
    description = pd.read_csv("description.csv")
    medications = pd.read_csv('medications.csv')
    diets = pd.read_csv("diets.csv")
    svc = pickle.load(open('svc.pkl', 'rb'))
except FileNotFoundError:
    print(
        "One or more of the required data files or model file (svc.pkl) were not found. Please ensure all files are in the same directory.")
    exit()

# ============================================================
# Helper Functions
# ============================================================

# Symptoms and diseases dictionaries
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}


def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else "No description available"

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre_list = pre.values.flatten().tolist() if not pre.empty else []
    pre_list = [item for item in pre_list if pd.notna(item) and str(item).strip() not in ('.', '')]
    pre = pre_list if pre_list else ["No precautions available"]

    # Use .tolist() to correctly get a list of items
    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist() if not med.empty else ["No medications available"]

    # Use .tolist() to correctly get a list of items
    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist() if not die.empty else ["No diet recommendations available"]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.tolist() if not wrkout.empty else ["No workout recommendations available"]

    return desc, pre, med, die, wrkout


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    # Predict the disease and return its name
    predicted_index = svc.predict([input_vector])[0]
    return diseases_list.get(predicted_index, "Unknown Disease")


# Database helper functions
def create_user(username, password):
    hashed_pw = generate_password_hash(password)
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()


def get_user(username):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user


def save_prediction(user_id, symptoms, predicted_disease):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (user_id, symptoms, predicted_disease) VALUES (?, ?, ?)",
              (user_id, ','.join(symptoms), predicted_disease))
    conn.commit()
    conn.close()


def get_user_predictions(user_id):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute(
        "SELECT id, symptoms, predicted_disease, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,))
    predictions = c.fetchall()
    conn.close()
    return predictions


# ============================================================
# Flask Routes
# ============================================================

@app.route("/")
def index():
    if 'user_id' in session:
        return render_template("index.html", logged_in=True, username=session['username'])
    return render_template("index.html", logged_in=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if create_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error="Username already exists")
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    predictions = get_user_predictions(session['user_id'])
    return render_template('history.html', predictions=predictions)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        if not symptoms or symptoms.strip() == "":
            message = "Please enter some symptoms."
            return render_template('index.html', message=message, logged_in='user_id' in session)

        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

        predicted_disease = get_predicted_value(user_symptoms)

        if predicted_disease == "Unknown Disease":
            message = "Could not predict a disease based on the symptoms provided. Please check for spelling errors."
            return render_template('index.html', message=message, logged_in='user_id' in session)

        # Call the helper function and store the results
        dis_des, my_precautions, medications, rec_diet, workout = helper(predicted_disease)

        # --- Add these print statements to your code ---
        print("Predicted Disease:", predicted_disease)
        print("Description:", dis_des)
        print("Precautions:", my_precautions)
        print("Medications:", medications)
        print("Recommended Diet:", rec_diet)
        print("Workout:", workout)
        # -----------------------------------------------

        # Save to DB if logged in
        if 'user_id' in session:
            save_prediction(session['user_id'], user_symptoms, predicted_disease)

        return render_template('index.html',
                               predicted_disease=predicted_disease,
                               dis_des=dis_des,
                               my_precautions=my_precautions,
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout,
                               logged_in='user_id' in session,
                               username=session.get('username'))

    return render_template('index.html', logged_in='user_id' in session)


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    app.run(debug=True)