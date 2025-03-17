import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

GROQ_API_KEY = "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"

model_chatbot = ChatGroq(model="Gemma2-9b-It", groq_api_key=GROQ_API_KEY)
parser = StrOutputParser()

# Load YOLO model
model = YOLO('.//best.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize buffers for angles and elbow positions
angle_buffer = deque(maxlen=10)
elbow_position_buffer = deque(maxlen=10)

# Global variables for rep counting and tracking
rep_count = 0
exercise_state = None  # Tracks the state of the exercise (e.g., "up" or "down")
prev_hip_y = None  # For push-up tracking

# Custom color themes
PRIMARY_COLOR = "#1F77B4"  # Blue
SECONDARY_COLOR = "#FF7F0E"  # Orange
SUCCESS_COLOR = "#2CA02C"  # Green
ERROR_COLOR = "#D62728"  # Red
BACKGROUND_COLOR = "#0047AB"  # Light gray

# Set page config for better UI
st.set_page_config(
    page_title="Posture Pal",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .stButton button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        font-weight: bold;
    }}
    .stMarkdown h1 {{
        color: {PRIMARY_COLOR};
    }}
    .stMarkdown h2 {{
        color: {SECONDARY_COLOR};
    }}
    .stProgress > div > div > div {{
        background-color: {SUCCESS_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# HTML content for the banner
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Posture Pal</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: Arial, sans-serif;
    }

    body {
      background-color: #e9f6e9;
      margin: 0;
      padding: 0;
    }

    .banner {
      background-image: url(https://static.vecteezy.com/system/resources/thumbnails/020/734/052/original/animated-studying-lo-fi-background-late-night-homework-2d-cartoon-character-animation-with-nighttime-cozy-bedroom-interior-on-background-4k-footage-with-alpha-channel-for-lofi-music-aesthetic-video.jpg);
      background-size: cover;
      background-position: center;
      width: 100%;
      height: 300px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      text-align: center;
    }

    .banner h1 {
      font-size: 2.5rem;
      margin: 0;
    }

    .banner h3 {
      font-size: 1.5rem;
      margin: 0;
    }

    .container-fluid {
      padding: 0 1rem;
      background-color: #f0f0f0;
    }

    .animation-container {
      animation: fly-in 1s ease-out both;
      opacity: 0;
      transform: translateX(-100%);
      text-align: center;
    }

    @keyframes fly-in {
      0% {
        transform: translateX(-100%);
        opacity: 0;
      }
      100% {
        transform: translateX(0);
        opacity: 1;
      }
    }

    h1, h2, h3 {
      font-family: 'Arial', sans-serif;
    }

    h1 {
      font-family: 'Pacifico', cursive;
      font-size: 2rem;
    }

    h2 {
      font-family: 'Lobster', cursive;
      font-size: 1.75rem;
    }

    h3 {
      font-family: 'Montserrat', sans-serif;
      font-size: 1.5rem;
    }

    p {
      font-family: 'Open Sans', sans-serif;
      font-size: 1rem;
    }
  </style>
</head>
<body>
  <div class="banner">
    <div>
      <h1>POSTURE PAL</h1>
      <h3><span id="typed-text"></span></h3>
    </div>
  </div>

  <script>
    document.addEventListener("DOMContentLoaded", function () {
      var typed = new Typed("#typed-text", {
        strings: [
          'Real-Time Exercise Feedback.',
          'Improve your posture with AI.',
          'Track your workouts effortlessly.',
          'Achieve your fitness goals.'
        ],
        typeSpeed: 50,
        backSpeed: 25,
        backDelay: 2000,
        startDelay: 1000,
        loop: true
      });
    });
  </script>
  <script src="https://unpkg.com/typed.js@2.0.16/dist/typed.umd.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
</body>
</html>
"""

# Display the banner
components.html(html_content, height=350)

# Introduction section
st.markdown("""
    ## Welcome to Posture Pal! 💪
    Posture Pal is your AI-powered fitness assistant that provides real-time feedback on your exercise form. 
    Whether you're doing bicep curls, push-ups, or squats, Posture Pal ensures you maintain proper posture 
    and maximize your workout efficiency.
""")

# Video source selection (moved to the main page)
st.markdown("---")
st.markdown("### 🎥 Select Video Source")
video_source = st.radio("Choose how you want to start:", ("Webcam", "Upload Video"))

if video_source == "Webcam":
    cap = cv2.VideoCapture(0)
elif video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_file:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot_product = np.dot(ab, bc)
    mag_ab = np.linalg.norm(ab)
    mag_bc = np.linalg.norm(bc)
    if mag_ab == 0 or mag_bc == 0:
        return 0
    angle = np.degrees(np.arccos(dot_product / (mag_ab * mag_bc)))
    return angle

# Function to check bicep curl form and count reps
def check_bicep_curl_form(landmarks, frame):
    global rep_count, exercise_state

    feedback = "Good Form"
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # Calculate elbow angle
    elbow_angle = calculate_angle(
        [left_shoulder[0], left_shoulder[1]],
        [left_elbow[0], left_elbow[1]],
        [left_wrist[0], left_wrist[1]]
    )

    # Check form
    shoulder_elbow_alignment = abs(left_shoulder[1] - left_elbow[1])
    if elbow_angle < 10 or elbow_angle > 170:
        feedback = "Bad Form: Incorrect Elbow Angle"
    if shoulder_elbow_alignment > 120:
        feedback = "Bad Form: Shoulder-Elbow Alignment Off"

    # Rep counting logic
    if elbow_angle > 160:  # Arm is extended
        exercise_state = "up"
    elif elbow_angle < 50 and exercise_state == "up":  # Arm is curled
        exercise_state = "down"
        rep_count += 1

    # Display feedback and rep count
    cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Shoulder-Elbow Alignment: {shoulder_elbow_alignment:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return feedback

# Function to check push-up form and count reps
def check_pushup_form(landmarks, frame):
    global rep_count, exercise_state, prev_hip_y

    feedback = "Good Form"
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

    # Calculate hip-shoulder distance for rep counting
    hip_shoulder_distance = abs(left_hip[1] - left_shoulder[1])

    # Rep counting logic
    if hip_shoulder_distance < 50:  # Body is lowered
        exercise_state = "down"
    elif hip_shoulder_distance > 100 and exercise_state == "down":  # Body is raised
        exercise_state = "up"
        rep_count += 1

    # Display feedback and rep count
    cv2.putText(frame, f"Hip-Shoulder Distance: {int(hip_shoulder_distance)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return feedback

# Function to check squat form and count reps
def check_squat_form(landmarks, frame):
    global rep_count, exercise_state

    feedback = "Good Form"
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    # Calculate knee angle
    knee_angle = calculate_angle(
        [left_hip[0], left_hip[1]],
        [left_knee[0], left_knee[1]],
        [left_ankle[0], left_ankle[1]]
    )

    # Rep counting logic
    if knee_angle < 90:  # Squatting position
        exercise_state = "down"
    elif knee_angle > 160 and exercise_state == "down":  # Standing position
        exercise_state = "up"
        rep_count += 1

    # Display feedback and rep count
    cv2.putText(frame, f"Knee Angle: {int(knee_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return feedback

# Start button
if st.button("Start Analysis"):
    if 'cap' in locals() and cap and cap.isOpened():
        st_frame = st.empty()
        progress_bar = st.progress(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect exercise type using YOLO
            results = model(frame)
            if len(results[0].boxes):
                sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
                class_id = int(sorted_boxes[0].cls)
                exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')
            else:
                exercise_type = 'Unknown'

            # Process frame with MediaPipe Pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            feedback = "Unknown"

            if pose_results.pose_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in pose_results.pose_landmarks.landmark]
                if exercise_type == "bicep curl":
                    feedback = check_bicep_curl_form(landmarks, frame)
                elif exercise_type == "push-up":
                    feedback = check_pushup_form(landmarks, frame)
                elif exercise_type == "squat":
                    feedback = check_squat_form(landmarks, frame)

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display exercise type and feedback
            cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color = (0, 255, 0) if "Good Form" in feedback else (0, 0, 255)
            cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display the frame in Streamlit
            st_frame.image(frame, channels="BGR", use_column_width=True)

            # Update progress bar and show motivational messages
            progress = min(rep_count / 10, 1.0)  # Assuming 10 reps as a goal
            progress_bar.progress(progress)
            if rep_count % 5 == 0 and rep_count > 0:
                st.success(np.random.choice([
                    "You're doing great! Keep it up! 💪",
                    "Every rep counts! 🔥",
                    "Stay strong and focused! 🚀",
                    "You're one step closer to your goal! 🌟"
                ]))

        cap.release()
    else:
        st.error("No video source selected or invalid file!")

# Input for user query
user_query = st.text_input("Enter your question or queries related to fitness:")

if user_query:
    # Create messages for the Groq model
    messages = [
        SystemMessage(content="You are a fitness bot. Provide detailed explanations, answer questions, and offer helpful resources related to the user's query on fitness and exercise related issues."),
        HumanMessage(content=user_query)
    ]
    
    # Get the response from the model
    try:
        response = parser.invoke(model_chatbot.invoke(messages))
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
