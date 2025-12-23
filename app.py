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
from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

app = Flask(__name__)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"


app = Flask(__name__)
app.secret_key = 'wellness123'
app.permanent_session_lifetime = timedelta(minutes=30)

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
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ======================
# Exercise Tracking (IN-MEMORY)
# ======================
exercise_states = {}

# ======================
# PDFKit Config
# ======================
config = pdfkit.configuration(
    wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
)

# ======================
# AUTH ROUTES (NO DATABASE)
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

        # --- Calculations ---
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

        # --- BMI Category ---
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

        # --- Save to JSON history ---
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
    
    # Reset the exercise state
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
# Pose Exercise Processing
# ======================
@app.route('/process_exercise_frame', methods=['POST'])
def process_exercise_frame():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    image_data = data['image']
    exercise_type = data['exercise_type']
    
    # Convert base64 image to OpenCV format
    image_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame with MediaPipe
    result = process_frame_with_mediapipe(frame, exercise_type, session['username'])
    
    return jsonify(result)

def process_frame_with_mediapipe(frame, exercise_type, username):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Exercise-specific rep counting logic
            reps, feedback, state = count_exercise_reps(results.pose_landmarks, exercise_type, username)
            
            # Convert processed image back to base64
            _, buffer = cv2.imencode('.jpg', image)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'reps': reps,
                'feedback': feedback,
                'landmarks_detected': True,
                'processed_image': f"data:image/jpeg;base64,{processed_image}",
                'state': state
            }
        
        return {
            'reps': 0,
            'feedback': 'No person detected - make sure you are visible in the camera',
            'landmarks_detected': False,
            'processed_image': None,
            'state': 'none'
        }

def count_exercise_reps(landmarks, exercise_type, username):
    user_key = f"{username}_{exercise_type}"
    
    if user_key not in exercise_states:
        exercise_states[user_key] = {
            'count': 0,
            'state': 'up',
            'prev_state': 'up',
            'threshold': 0.2
        }
    
    state = exercise_states[user_key]
    
    if exercise_type == 'pushups':
        return count_pushups(landmarks, state)
    elif exercise_type == 'squats':
        return count_squats(landmarks, state)
    elif exercise_type == 'bicep_curls':
        return count_bicep_curls(landmarks, state)
    else:
        return state['count'], "Exercise type not supported", state['state']

def count_pushups(landmarks, state):
    # Use shoulder, elbow, and wrist for pushup detection
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    
    # Calculate vertical movement (simplified)
    shoulder_y = left_shoulder.y
    wrist_y = left_wrist.y
    vertical_diff = abs(shoulder_y - wrist_y)
    
    if vertical_diff < 0.15 and state['state'] == 'down':
        state['count'] += 1
        state['state'] = 'up'
        feedback = f"Pushup #{state['count']} completed!"
    elif vertical_diff > 0.25:
        state['state'] = 'down'
        feedback = "Go lower for a complete pushup"
    else:
        feedback = "Maintain form"
    
    return state['count'], feedback, state['state']

def count_squats(landmarks, state):
    # Use hip, knee, and ankle for squat detection
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    
    # Calculate knee angle
    hip_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    
    if hip_knee_angle > 160 and state['state'] == 'down':
        state['count'] += 1
        state['state'] = 'up'
        feedback = f"Squat #{state['count']} completed!"
    elif hip_knee_angle < 100:
        state['state'] = 'down'
        feedback = "Good depth! Come back up"
    else:
        feedback = "Lower yourself into a squat"
    
    return state['count'], feedback, state['state']

def count_bicep_curls(landmarks, state):
    # Use shoulder, elbow, and wrist for bicep curls
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    
    # Calculate angle at elbow
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    if elbow_angle > 160 and state['state'] == 'up':
        state['state'] = 'down'
        feedback = "Curl up now"
    elif elbow_angle < 50 and state['state'] == 'down':
        state['count'] += 1
        state['state'] = 'up'
        feedback = f"Bicep curl #{state['count']} completed!"
    else:
        feedback = "Continue curling"
    
    return state['count'], feedback, state['state']

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# ======================
# Hand Exercise Processing
# ======================
@app.route('/process_hand_frame', methods=['POST'])
def process_hand_frame():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    image_data = data['image']
    
    # Convert base64 image to OpenCV format
    image_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame with MediaPipe Hands
    result = process_hand_frame_with_mediapipe(frame, session['username'])
    
    return jsonify(result)

def process_hand_frame_with_mediapipe(frame, username):
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = hands.process(image)
        
        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        hand_landmarks_list = []
        handedness_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                hand_landmarks_list.append(hand_landmarks)
                handedness_list.append(handedness.classification[0].label)
            
            # Count hand reps
            reps, feedback, state, hand_states = count_hand_reps(hand_landmarks_list, handedness_list, username)
            
            # Convert processed image back to base64
            _, buffer = cv2.imencode('.jpg', image)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'reps': reps,
                'feedback': feedback,
                'hands_detected': True,
                'processed_image': f"data:image/jpeg;base64,{processed_image}",
                'state': state,
                'hand_states': hand_states
            }
        
        return {
            'reps': 0,
            'feedback': 'No hands detected - show your hands to the camera',
            'hands_detected': False,
            'processed_image': None,
            'state': 'none',
            'hand_states': {}
        }

def count_hand_reps(hand_landmarks_list, handedness_list, username):
    user_key = f"{username}_hand_workout"
    
    if user_key not in exercise_states:
        exercise_states[user_key] = {
            'count': 0,
            'state': 'open',
            'prev_state': 'open',
            'threshold': 0.5
        }
    
    state = exercise_states[user_key]
    hand_states = {}
    
    if not hand_landmarks_list:
        return state['count'], "Show your hands to the camera", state['state'], hand_states
    
    # Analyze each hand
    for i, (hand_landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
        is_open = is_hand_open(hand_landmarks)
        hand_states[handedness] = 'open' if is_open else 'closed'
    
    # Determine overall state (use first hand or combine logic)
    if hand_landmarks_list:
        main_hand_open = is_hand_open(hand_landmarks_list[0])
        current_state = 'open' if main_hand_open else 'closed'
        
        # Count reps: open → closed → open OR closed → open → closed
        if current_state != state['prev_state']:
            if state['prev_state'] != state['state']:  # We've completed a cycle
                state['count'] += 1
                state['state'] = current_state
                feedback = f"Rep #{state['count']} completed! Hand is {current_state}"
            else:
                feedback = f"Hand is {current_state} - continue the motion"
            
            state['prev_state'] = current_state
        else:
            feedback = f"Keep going - hand is {current_state}"
    else:
        feedback = "No hand state detected"
        current_state = 'none'
    
    return state['count'], feedback, current_state, hand_states

def is_hand_open(hand_landmarks):
    """
    Determine if hand is open or closed based on finger positions
    Returns True if hand is open, False if closed
    """
    # Get key landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Calculate distances from finger tips to their MCP joints
    def distance(landmark1, landmark2):
        return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5
    
    # Check if fingers are extended (open)
    index_extended = distance(index_tip, index_mcp) > distance(thumb_tip, thumb_mcp) * 1.5
    middle_extended = distance(middle_tip, middle_mcp) > distance(thumb_tip, thumb_mcp) * 1.5
    ring_extended = distance(ring_tip, index_mcp) > distance(thumb_tip, thumb_mcp) * 1.2
    pinky_extended = distance(pinky_tip, index_mcp) > distance(thumb_tip, thumb_mcp) * 1.2
    
    # Simple logic: hand is open if most fingers are extended
    extended_fingers = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    
    return extended_fingers >= 3


# ======================
# Save Exercise Progress (JSON-based)
# ======================
# ======================
# Save Exercise Progress (JSON-based)
# ======================
@app.route('/save_exercise_progress', methods=['POST'])
def save_exercise_progress():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.json
    username = session['username']

    # Load existing exercise history
    exercise_history = load_history(username)  # reuse the same JSON file

    # Format the date consistently
    workout_date = data.get('date')
    if workout_date:
        # If client sends a date, try to parse and reformat it
        try:
            workout_date = datetime.strptime(workout_date, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M")
        except:
            # fallback to current time
            workout_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        workout_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Append new entry
    entry = {
        'date': workout_date,
        'exercise_type': data['exercise_type'],
        'reps': data['reps']
    }
    exercise_history.append(entry)

    # Save back to JSON
    save_history(username, exercise_history)

    return jsonify({'success': True, 'message': 'Exercise progress saved!'})


# ======================
# Exercise History (JSON-based)
# ======================
@app.route('/exercise_history')
def exercise_history():
    if 'username' not in session:
        return redirect('/login')
    
    username = session['username']
    
    # Load exercise history from JSON
    records = load_history(username)
    
    # Filter only exercise entries
    exercise_records = [r for r in records if 'exercise_type' in r]
    
    # Sort by date descending
    exercise_records = sorted(exercise_records, key=lambda x: x['date'], reverse=True)
    
    return render_template('exercise_history.html', records=exercise_records)



# ======================
# Other Routes
# ======================

# Route to render the diet plan page
@app.route("/diet_plan")
def diet_plan():
    return render_template("diet_plan.html")

# Backend route to generate diet plan
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
        data = response.json()
        return jsonify(data)
    else:
        return jsonify({"error": "Failed to generate diet plan"}), 500
@app.route('/health_info')
def health_info():
    if 'username' not in session:
        return redirect('/login')
    return render_template('health_info.html')

@app.route('/measure_height')
def measure_height():
    return render_template('measure_height.html')

@app.route("/recipes", methods=["GET", "POST"])
def recipes():
    if 'username' not in session:
        return redirect('/login')

    recipe = None  # keep for future use if needed

    # Load API key from .env
    api_key = os.getenv("OPENROUTER_API_KEY")

    if request.method == "POST":
        dish_name = request.form.get("dish_name")
        if dish_name:
            # Placeholder for future recipe logic
            pass

    return render_template(
        "recipe_chat.html",
        recipe=recipe,
        api_key=api_key
    )

# ======================
# Challenges
# ======================


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

        return redirect(url_for('challenge'))  # reload page to show updated history

    return render_template('challenge.html', challenges=challenges)


# ======================
# Run App
# ======================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

