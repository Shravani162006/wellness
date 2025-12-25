from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from datetime import timedelta, datetime
import os, json, requests
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
    file = get_user_file(username)
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return []

def save_history(username, history):
    with open(get_user_file(username), "w") as f:
        json.dump(history, f, indent=4)

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
        if not username:
            return render_template("index.html", error="Enter username")
        session["username"] = username
        return redirect("/health")
    return render_template("index.html")

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
        gender = request.form["gender"]
        age = int(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])

        h_m = height / 100
        bmi = round(weight / (h_m ** 2), 2)

        if gender.lower() == "male":
            bmr = round(10 * weight + 6.25 * height - 5 * age + 5, 2)
        else:
            bmr = round(10 * weight + 6.25 * height - 5 * age - 161, 2)

        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bmi": bmi,
            "bmr": bmr,
            "height": height,
            "weight": weight,
            "age": age,
            "gender": gender
        }

        history = load_history(session["username"])
        history.append(entry)
        save_history(session["username"], history)

        session["last_result"] = entry
        flash("Health data saved!")
        return redirect("/result")

    return render_template("health_form.html")

@app.route("/result")
def result():
    if "last_result" not in session:
        return redirect("/health")
    return render_template("result.html", **session["last_result"])

# ======================
# HISTORY
# ======================
@app.route("/history")
def history():
    if "username" not in session:
        return redirect("/login")
    records = load_history(session["username"])
    return render_template("history.html", records=records)

# ======================
# EXERCISES
# ======================
@app.route("/exercise")
def exercise():
    if "username" not in session:
        return redirect("/login")
    return render_template("exercise.html")

@app.route("/exercise/hand_workout")
def hand_workout():
    if "username" not in session:
        return redirect("/login")
    return render_template("hand_workout.html")

@app.route("/exercise/<exercise_type>")
def other_exercise(exercise_type):
    if "username" not in session:
        return redirect("/login")
    return render_template("exercise_tracker.html", exercise_type=exercise_type)

# ======================
# SAVE EXERCISE PROGRESS
# ======================
@app.route("/save_exercise_progress", methods=["POST"])
def save_exercise_progress():
    if "username" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    history = load_history(session["username"])

    entry = {
        "date": data.get("date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "exercise_type": data["exercise_type"],
        "reps": data["reps"]
    }

    history.append(entry)
    save_history(session["username"], history)

    return jsonify({"success": True})

@app.route("/exercise_history")
def exercise_history():
    if "username" not in session:
        return redirect("/login")

    records = load_history(session["username"])
    exercises = [r for r in records if "exercise_type" in r]
    return render_template("exercise_history.html", records=exercises)

# ======================
# DIET PLAN
# ======================
@app.route("/diet_plan")
def diet_plan():
    return render_template("diet_plan.html")

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    data = request.json
    report = data.get("report", "")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a professional dietitian."},
            {"role": "user", "content": report}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return jsonify(response.json())

# ======================
# OTHER PAGES
# ======================
@app.route("/health_info")
def health_info():
    return render_template("health_info.html")

@app.route("/measure_height")
def measure_height():
    return render_template("measure_height.html")

@app.route("/recipes")
def recipes():
    return render_template("recipe_chat.html", api_key=API_KEY)

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)
