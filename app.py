from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from datetime import timedelta, datetime
import os, uuid, json, base64
import numpy as np
import cv2
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "wellness123"
app.permanent_session_lifetime = timedelta(minutes=30)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ======================
# User Data Storage
# ======================
USER_DATA_FOLDER = "user_data"
os.makedirs(USER_DATA_FOLDER, exist_ok=True)

def get_user_file(username):
    return os.path.join(USER_DATA_FOLDER, f"{username}.json")

def load_history(username):
    if os.path.exists(get_user_file(username)):
        with open(get_user_file(username)) as f:
            return json.load(f)
    return []

def save_history(username, history):
    with open(get_user_file(username), "w") as f:
        json.dump(history, f, indent=4)

# ======================
# MediaPipe Setup (SAFE)
# ======================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("MediaPipe disabled:", e)
    MEDIAPIPE_AVAILABLE = False

# ======================
# Exercise States
# ======================
exercise_states = {}

# ======================
# AUTH
# ======================
@app.route('/')
def home():
    return redirect('/login')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect('/health')
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form['username']:
            session['username'] = request.form['username']
            return redirect('/health')
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ======================
# HEALTH
# ======================
@app.route('/health', methods=['GET','POST'])
def health():
    if 'username' not in session:
        return redirect('/login')

    if request.method == 'POST':
        gender = request.form['gender']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        h_m = height / 100
        bmi = round(weight / (h_m ** 2), 2)

        if gender.lower() == 'male':
            bmr = round(10*weight + 6.25*height - 5*age + 5, 2)
            mm = round(weight * 0.45, 2)
        else:
            bmr = round(10*weight + 6.25*height - 5*age - 161, 2)
            mm = round(weight * 0.41, 2)

        ideal_weight = round(21.7 * (h_m ** 2), 2)
        suggestion = "Perfect Weight"
        if weight < ideal_weight: suggestion = "Gain Weight"
        if weight > ideal_weight: suggestion = "Lose Weight"

        bmi_category = "Normal"
        health_effects = "Healthy range"

        result = {
            "bmi": bmi, "bmr": bmr, "mm": mm,
            "ideal_weight": ideal_weight,
            "suggestion": suggestion,
            "bmi_category": bmi_category,
            "health_effects": health_effects
        }

        history = load_history(session['username'])
        history.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), **result})
        save_history(session['username'], history)

        session['last_result'] = result
        return redirect(url_for('result'))

    return render_template('health_form.html')

@app.route('/result')
def result():
    if 'last_result' not in session:
        return redirect('/health')
    return render_template('result.html', **session['last_result'])

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect('/login')
    records = load_history(session['username'])
    return render_template('history.html', records=records)

@app.route('/download_history')
def download_history():
    flash("PDF download is disabled on deployed server.")
    return redirect('/history')

# ======================
# EXERCISE
# ======================
@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/exercise/<exercise_type>')
def specific_exercise(exercise_type):
    return render_template(
        'hand_workout.html' if exercise_type == 'hand_workout'
        else 'exercise_tracker.html',
        exercise_type=exercise_type
    )

@app.route('/reset_exercise', methods=['POST'])
def reset_exercise():
    data = request.json
    key = f"{session['username']}_{data['exercise_type']}"
    exercise_states[key] = {'count':0,'prev_state':'up','state':'up'}
    return jsonify(success=True)

# ======================
# HAND WORKOUT (FIXED)
# ======================
def is_hand_open(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    return sum(
        hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y
        for t,p in zip(tips,pips)
    ) >= 3

def process_hand_frame_with_mediapipe(frame, username):
    key = f"{username}_hand_workout"
    if key not in exercise_states:
        exercise_states[key] = {'count':0,'prev':{'Left':'open','Right':'open'}}

    with mp_hands.Hands(max_num_hands=2) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hd.classification[0].label
                state = 'open' if is_hand_open(lm) else 'closed'
                if exercise_states[key]['prev'][label]=='closed' and state=='open':
                    exercise_states[key]['count'] += 1
                exercise_states[key]['prev'][label] = state

                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        _, buf = cv2.imencode('.jpg', frame)
        return {
            "reps": exercise_states[key]['count'],
            "processed_image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        }

@app.route('/process_hand_frame', methods=['POST'])
def process_hand_frame():
    if not MEDIAPIPE_AVAILABLE:
        return jsonify(error="Hand tracking not supported"),503

    img = request.json['image'].split(',')[1]
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(img),np.uint8),1)
    return jsonify(process_hand_frame_with_mediapipe(frame, session['username']))

# ======================
# DIET / INFO / OTHER
# ======================
@app.route('/diet_plan')
def diet_plan():
    return render_template('diet_plan.html')

@app.route('/health_info')
def health_info():
    return render_template('health_info.html')

@app.route('/measure_height')
def measure_height():
    return render_template('measure_height.html')

@app.route('/recipes')
def recipes():
    return render_template('recipe_chat.html')

@app.route('/challenge')
def challenge():
    return render_template('challenge.html')

# ======================
# RUN
# ======================
if __name__ == '__main__':
    app.run(debug=True)
