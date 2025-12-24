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

load_dotenv()  # Load .env variables

app = Flask(__name__)
app.secret_key = 'wellness123'
app.permanent_session_lifetime = timedelta(minutes=30)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ======================
# Folder to store user JSON files
# ======================
USER_DATA_FOLDER = "user_data"
os.makedirs(USER_DATA_FOLDER, exist_ok=True)

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
    filepath = get_user_file(username)
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)

# ======================
# MediaPipe Setup
# ======================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("Mediapipe disabled:", e)
    MEDIAPIPE_AVAILABLE = False

# ======================
# Exercise Tracking (IN-MEMORY)
# ======================
exercise_states = {}

# ======================
# PDFKit Config
# ======================
try:
    config = pdfkit.configuration()
except:
    config = None

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
        else:
            return render_template('index.html', error="Please enter username")
    return render_template('index.html')

# ======================
# Health Calculator Route
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

# ======================
# History & Download Routes
# ======================
@app.route('/history')
def history():
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
        return redirect(url_for('history'))

    rendered = render_template('history_pdf.html', records=records)
    output_filename = f"health_history_{uuid.uuid4()}.pdf"
    output_path = os.path.join(os.getcwd(), output_filename)

    if not config:
        flash("PDF download not available on server")
        return redirect(url_for('history'))

    pdfkit.from_string(rendered, output_path, configuration=config, options={"enable-local-file-access": True})
    response = send_file(output_path, as_attachment=True)

    @response.call_on_close
    def cleanup():
        try:
            os.remove(output_path)
        except:
            pass

    return response

# ======================
# Logout
# ======================
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# ======================
# Exercise Routes
# ======================
@app.route('/exercise')
def exercise():
    if 'username' not in session:
        return redirect('/login')
    return render_template('exercise.html')

@app.route('/exercise/<exercise_type>')
def specific_exercise(exercise_type):
    if 'username' not in session:
        return redirect('/login')
    if exercise_type == 'hand_workout':
        return render_template('hand_workout.html')
    else:
        return render_template('exercise_tracker.html', exercise_type=exercise_type)

@app.route('/exercise/hand_workout')
def hand_workout():
    if 'username' not in session:
        return redirect('/login')
    return render_template('hand_workout.html')

# ======================
# Reset Exercise Route
# ======================
@app.route('/reset_exercise', methods=['POST'])
def reset_exercise():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    exercise_type = data['exercise_type']
    username = session['username']

    user_key = f"{username}_{exercise_type}"
    if user_key in exercise_states:
        exercise_states[user_key] = {
            'count': 0,
            'state': 'up',
            'prev_state': 'up',
            'threshold': 0.2
        }

    return jsonify({'success': True, 'message': 'Exercise counter reset to 0'})

# ======================
# ======================
# ====== NEW FUNCTIONS ======
# ======================
# ======================

# ---- Pose Exercise ----
def process_frame_with_mediapipe(frame, exercise_type, username):
    results = {}
    user_key = f"{username}_{exercise_type}"

    if user_key not in exercise_states:
        exercise_states[user_key] = {'count': 0, 'state': 'up', 'prev_state': 'up'}

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Simple bicep curl logic example
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            if left_wrist.y < left_elbow.y:
                exercise_states[user_key]['state'] = 'up'
            else:
                exercise_states[user_key]['state'] = 'down'

            if exercise_states[user_key]['prev_state'] == 'down' and exercise_states[user_key]['state'] == 'up':
                exercise_states[user_key]['count'] += 1

            exercise_states[user_key]['prev_state'] = exercise_states[user_key]['state']

        results['count'] = exercise_states[user_key]['count']
        _, buffer = cv2.imencode('.jpg', frame)
        results['processed_image'] = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()

    return results

# ---- Hand Exercise ----
def process_hand_frame_with_mediapipe(frame, username):
    results = {}
    user_key = f"{username}_hand"

    if user_key not in exercise_states:
        exercise_states[user_key] = {'count': 0, 'state': 'open', 'prev_state': 'open', 'threshold': 0.5}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image_rgb)

        hand_states = {}

        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Simple open/closed detection
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                distance = lambda a, b: ((a.x-b.x)**2 + (a.y-b.y)**2)**0.5
                is_open = distance(index_tip, index_mcp) > distance(thumb_tip, index_mcp) * 1.5

                hand_states[handedness.classification[0].label] = 'open' if is_open else 'closed'

            # Count reps: first hand only
            main_hand_open = list(hand_states.values())[0] == 'open'
            current_state = 'open' if main_hand_open else 'closed'

            if current_state != exercise_states[user_key]['prev_state']:
                exercise_states[user_key]['count'] += 1
                exercise_states[user_key]['prev_state'] = current_state

        results['count'] = exercise_states[user_key]['count']
        results['hand_states'] = hand_states
        _, buffer = cv2.imencode('.jpg', frame)
        results['processed_image'] = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode()

    return results

# ======================
# The rest of your routes remain exactly the same
# ======================

# (Keep all existing save_exercise_progress, diet_plan, challenge, recipes, etc.)

# ======================
# Run App
# ======================
if __name__ == '__main__':
    app.run(debug=True)
