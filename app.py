from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from datetime import timedelta, datetime
import os, json, base64, uuid
import numpy as np
import cv2
import requests
from dotenv import load_dotenv

load_dotenv()

# ======================
# App Config
# ======================
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

def save_history(username, data):
    with open(get_user_file(username), "w") as f:
        json.dump(data, f, indent=4)

# ======================
# MediaPipe (SAFE)
# ======================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("❌ MediaPipe disabled:", e)
    MEDIAPIPE_AVAILABLE = False

exercise_states = {}

# ======================
# AUTH
# ======================
@app.route("/")
def home():
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        if username:
            session["username"] = username
            return redirect("/health")
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        session["username"] = request.form.get("username")
        return redirect("/health")
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ======================
# HEALTH
# ======================
@app.route("/health", methods=["GET", "POST"])
def health():
    if "username" not in session:
        return redirect("/login")

    if request.method == "POST":
        age = int(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        gender = request.form["gender"]

        bmi = round(weight / ((height/100)**2), 2)
        bmr = round(10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161), 2)

        result = {
            "bmi": bmi,
            "bmr": bmr,
            "tsf": bmi * 0.8,
            "vf": "Normal",
            "mm": weight * 0.45,
            "ideal_weight": round(21.7*((height/100)**2), 2),
            "suggestion": "Healthy",
            "bmi_category": "Normal",
            "health_effects": "Good health"
        }

        session["last_result"] = result
        history = load_history(session["username"])
        history.append({**result, "date": datetime.now().strftime("%Y-%m-%d %H:%M")})
        save_history(session["username"], history)

        return redirect("/result")

    return render_template("health_form.html")

@app.route("/result")
def result():
    if "last_result" not in session:
        return redirect("/health")
    return render_template("result.html", **session["last_result"])

@app.route("/history")
def history():
    if "username" not in session:
        return redirect("/login")
    return render_template("history.html", records=load_history(session["username"]))

@app.route("/download_history")
def download_history():
    flash("PDF download is disabled on server.")
    return redirect("/history")

# ======================
# EXERCISE PAGES
# ======================
@app.route("/exercise")
def exercise():
    return render_template("exercise.html")

@app.route("/exercise/<exercise_type>")
def exercise_page(exercise_type):
    if exercise_type == "hand_workout":
        return render_template("hand_workout.html")
    return render_template("exercise_tracker.html", exercise_type=exercise_type)

# ======================
# RESET EXERCISE
# ======================
@app.route("/reset_exercise", methods=["POST"])
def reset_exercise():
    exercise_states.clear()
    return jsonify(success=True)

# ======================
# EXERCISE PROCESS (SAFE)
# ======================
@app.route("/process_exercise_frame", methods=["POST"])
def process_exercise_frame():
    if not MEDIAPIPE_AVAILABLE:
        return jsonify(error="Exercise tracking not supported on server"), 503
    return jsonify(count=0, processed_image="")

@app.route("/process_hand_frame", methods=["POST"])
def process_hand_frame():
    if not MEDIAPIPE_AVAILABLE:
        return jsonify(error="Hand tracking not supported on server"), 503
    return jsonify(reps=0, processed_image="")

# ======================
# SAVE EXERCISE
# ======================
@app.route("/save_exercise_progress", methods=["POST"])
def save_exercise_progress():
    data = request.json
    history = load_history(session["username"])
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exercise_type": data["exercise_type"],
        "reps": data["reps"]
    })
    save_history(session["username"], history)
    return jsonify(success=True)

@app.route("/exercise_history")
def exercise_history():
    return render_template("exercise_history.html", records=load_history(session["username"]))

# ======================
# DIET / OTHER
# ======================
@app.route("/diet_plan")
def diet_plan():
    return render_template("diet_plan.html")

@app.route("/health_info")
def health_info():
    return render_template("health_info.html")

@app.route("/recipes")
def recipes():
    return render_template("recipe_chat.html")

@app.route("/challenge", methods=["GET", "POST"])
def challenge():
    return render_template("challenge.html")

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)
