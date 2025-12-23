from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from datetime import timedelta, datetime
import pdfkit
import os
import uuid
import json
import cv2
import mediapipe as mp
import numpy as np
import base64
import requests
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

app = Flask(__name__)
app.secret_key = 'wellness123'
app.permanent_session_lifetime = timedelta(minutes=30)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ----------------------
# Folder to store user JSON files
# ----------------------
USER_DATA_FOLDER = "user_data"
os.makedirs(USER_DATA_FOLDER, exist_ok=True)

# ----------------------
# MediaPipe Setup
# ----------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Exercise tracking (in-memory)
exercise_states = {}

# PDFKit configuration
config = pdfkit.configuration(
    wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'  # Adjust path for server
)

# ----------------------
# Helper functions for user data
# ----------------------
def get_user_file(username):
    return os.path.join(USER_DATA_FOLDER, f"{username}.json")

def load_history(username):
    filepath = get_user_file(username)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_history(username, history):
    filepath = get_user_file(username)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)

# ----------------------
# AUTH ROUTES
# ----------------------
@app.route('/')
def home():
    return redirect('/login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect('/health')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        if username:
            session['username'] = username
            return redirect('/health')
        else:
            return render_template('index.html', error="Please enter username")
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# ----------------------
# Health Calculator
# ----------------------
@app.route('/health', methods=['GET', 'POST'])
def health():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    history = load_history(username)

    if request.method == 'POST':
        gender = request.form['gender']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)

        if gender.lower() == 'male':
            bmr = round(10 * weight + 6.25 * height - 5 * age + 5, 2)
            mm = round(weight * 0.45, 2)
        else:
            bmr = round(10 * weight + 6.25 * height - 5 * age - 161, 2)
            mm = round(weight * 0.41, 2)

        ideal_bmi = 21.7
        ideal_weight = round(ideal_bmi * (height_m ** 2), 2)

        suggestion = (
            "Gain Weight" if weight < ideal_weight
            else "Lose Weight" if weight > ideal_weight
            else "Perfect Weight"
        )

        tsf = round((bmi * 0.8) + (age * 0.1), 2)

        if bmi < 18.5:
            vf = "Low"
        elif bmi <= 24.9:
            vf = "Normal"
        elif bmi <= 29.9:
            vf = "High"
        else:
            vf = "Very High"

        # BMI category
        if bmi < 18:
            bmi_category = "Malnutrition 2"
            health_effects = "Anorexia, Bulimia, Breakdown of muscle mass etc."
        elif bmi <= 20.0:
            bmi_category = "Malnutrition 1"
            health_effects = "Digestive problems, Weakness, Stress, Anxiety, Reproductive issues"
        elif bmi <= 23.0:
            bmi_category = "Normal"
            health_effects = "Normal menstruation, Energy, Vitality, Good resistance to illness"
        elif bmi <= 25.0:
            bmi_category = "Overweight"
            health_effects = "Fatigue, Digestive problems, Circulation issues, Varicose veins"
        elif bmi <= 28.0:
            bmi_category = "Obesity Grade 1"
            health_effects = "Diabetes, Hypertension, Joint problems (spine, knees), Strokes"
        elif bmi <= 30.0:
            bmi_category = "Obesity Grade 2"
            health_effects = "Diabetes, Cancer, Arthritis, Arteriosclerosis, Heart Attacks"
        else:
            bmi_category = "Obesity Grade 3"
            health_effects = "Max risk of Diabetes, Heart Disease, Cancer, Premature Death"

        # Save history
        entry = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'gender': gender,
            'age': age,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'bmr': bmr,
            'tsf': tsf,
            'vf': vf,
            'mm': mm,
            'ideal_weight': ideal_weight,
            'suggestion': suggestion,
            'bmi_category': bmi_category,
            'health_effects': health_effects
        }
        history.append(entry)
        save_history(username, history)
        flash("Health data saved successfully!")

        return render_template(
            'result.html',
            bmi=bmi,
            bmr=bmr,
            tsf=tsf,
            vf=vf,
            mm=mm,
            ideal_weight=ideal_weight,
            suggestion=suggestion,
            bmi_category=bmi_category,
            health_effects=health_effects
        )

    return render_template('health_form.html')

# ----------------------
# History & PDF Download
# ----------------------
@app.route('/history')
def history_page():
    if 'username' not in session:
        return redirect('/login')
    username = session['username']
    records = sorted(load_history(username), key=lambda x: x['date'], reverse=True)
    return render_template('history.html', records=records)

@app.route('/download_history')
def download_history():
    if 'username' not in session:
        return redirect('/login')

    username = session['username']
    records = sorted(load_history(username), key=lambda x: x['date'], reverse=True)
    if not records:
        flash("No history to download!")
        return redirect(url_for('history_page'))

    rendered = render_template('history_pdf.html', records=records)
    output_filename = f"health_history_{uuid.uuid4()}.pdf"
    output_path = os.path.join(os.getcwd(), output_filename)
    pdfkit.from_string(rendered, output_path, configuration=config, options={"enable-local-file-access": True})
    response = send_file(output_path, as_attachment=True)

    @response.call_on_close
    def cleanup():
        try:
            os.remove(output_path)
        except:
            pass

    return response

# ----------------------
# Exercise Routes & MediaPipe Processing
# ----------------------
@app.route('/exercise')
def exercise():
    if 'username' not in session:
        return redirect('/login')
    return render_template('exercise.html')

# You already have all the detailed exercise processing functions here
# pushups, squats, bicep curls, hand workout
# process_frame_with_mediapipe, process_hand_frame_with_mediapipe
# count_exercise_reps, count_pushups, count_squats, count_bicep_curls, is_hand_open
# calculate_angle
# save_exercise_progress, exercise_history

# ----------------------
# Diet Plan
# ----------------------
@app.route("/diet_plan")
def diet_plan():
    if 'username' not in session:
        return redirect('/login')
    return render_template("diet_plan.html")

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    data = request.get_json()
    report_text = data.get("report", "")

    prompt = f"""
    You are Wellora, a certified nutrition AI. Based on the following health report, create a 
    personalized 7-day diet plan that supports healthy weight loss, balanced nutrition, and 
    energy maintenance. Include Indian food options and calorie suggestions per meal.

    Health Report:
    {report_text}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a professional dietitian AI named Wellora."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.ok:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to generate diet plan"}), 500

# ----------------------
# Recipes
# ----------------------
@app.route("/recipes", methods=["GET", "POST"])
def recipes():
    if 'username' not in session:
        return redirect('/login')
    api_key = os.getenv("OPENROUTER_API_KEY")
    return render_template("recipe_chat.html", recipe=None, api_key=api_key)

# ----------------------
# Challenges
# ----------------------
challenges = []

@app.route('/challenge', methods=['GET', 'POST'])
def challenge():
    if request.method == 'POST':
        challenge_name = request.form['challenge_name']
        goal = request.form['goal']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        total_days = request.form['total_days']

        challenges.append({
            'challenge_name': challenge_name,
            'goal': goal,
            'start_date': start_date,
            'end_date': end_date,
            'total_days': total_days
        })

        return redirect(url_for('challenge'))

    return render_template('challenge.html', challenges=challenges)

# ----------------------
# Run App
# ----------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
