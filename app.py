import sys
print(sys.version)

from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from datetime import timedelta, datetime
import pdfkit
import os
import uuid
import json
import cv2
import numpy as np
import base64
import requests
from dotenv import load_dotenv

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()

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
# SAFE MEDIAPIPE IMPORT (RENDER FIX)
# ----------------------
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("Mediapipe failed to load:", e)
    mp_pose = None
    mp_hands = None
    mp_drawing = None
    mp_drawing_styles = None
    MEDIAPIPE_AVAILABLE = False

# ----------------------
# Exercise tracking (in-memory)
# ----------------------
exercise_states = {}

# ----------------------
# PDFKit configuration (Render-safe)
# ----------------------
WKHTML_PATH = os.getenv("WKHTMLTOPDF_PATH")
config = pdfkit.configuration(wkhtmltopdf=WKHTML_PATH) if WKHTML_PATH else None

# ======================
# Helper functions
# ======================
def get_user_file(username):
    return os.path.join(USER_DATA_FOLDER, f"{username}.json")

def load_history(username):
    filepath = get_user_file(username)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_history(username, history):
    with open(get_user_file(username), "w") as f:
        json.dump(history, f, indent=4)

# ======================
# Exercise counting stubs
# ======================
def count_exercise_reps(pose_landmarks, exercise_type, username):
    return 0, f"{exercise_type} counting not implemented yet.", "none"

def count_hand_reps(hand_landmarks_list, handedness_list, username):
    hand_states = {hand: "unknown" for hand in handedness_list}
    return 0, "Hand exercise counting not implemented yet.", "none", hand_states

# ======================
# AUTH ROUTES
# ======================
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
        return render_template('index.html', error="Please enter username")
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# ======================
# HEALTH CALCULATOR
# ======================
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

        ideal_weight = round(21.7 * (height_m ** 2), 2)
        suggestion = "Gain Weight" if weight < ideal_weight else "Lose Weight" if weight > ideal_weight else "Perfect Weight"

        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "gender": gender,
            "age": age,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "bmr": bmr,
            "mm": mm,
            "ideal_weight": ideal_weight,
            "suggestion": suggestion
        }

        history.append(entry)
        save_history(username, history)
        flash("Health data saved successfully!")
        return render_template("result.html", **entry)

    return render_template("health_form.html")

# ======================
# HISTORY & PDF
# ======================
@app.route('/history')
def history_page():
    if 'username' not in session:
        return redirect('/login')
    records = sorted(load_history(session['username']), key=lambda x: x['date'], reverse=True)
    return render_template('history.html', records=records)

@app.route('/download_history')
def download_history():
    if 'username' not in session:
        return redirect('/login')

    records = load_history(session['username'])
    if not records:
        flash("No history to download!")
        return redirect(url_for('history_page'))

    rendered = render_template('history_pdf.html', records=records)
    output_path = f"/tmp/{uuid.uuid4()}.pdf"
    pdfkit.from_string(rendered, output_path, configuration=config)
    return send_file(output_path, as_attachment=True)

# ======================
# DIET PLAN
# ======================
@app.route("/diet_plan")
def diet_plan():
    return render_template("diet_plan.html")

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    data = request.get_json()
    prompt = f"Create a 7-day Indian diet plan.\n{data.get('report','')}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"model": "openai/gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
    res = requests.post(API_URL, headers=headers, json=payload)
    return jsonify(res.json())

# ======================
# EXERCISE ROUTES (SAFE)
# ======================
@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/process_exercise_frame', methods=['POST'])
def process_exercise_frame():
    if not MEDIAPIPE_AVAILABLE:
        return jsonify({"error": "Pose detection unavailable on server"}), 503
    return jsonify({"message": "Pose processing disabled on Render"})

@app.route('/process_hand_frame', methods=['POST'])
def process_hand_frame():
    if not MEDIAPIPE_AVAILABLE:
        return jsonify({"error": "Hand detection unavailable on server"}), 503
    return jsonify({"message": "Hand processing disabled on Render"})

# ======================
# RUN APP
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
