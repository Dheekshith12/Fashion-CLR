from flask import Flask, request, render_template, jsonify
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# Load Dlib Face Detector and Landmark Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect skin tone
def get_skin_tone(image, landmarks):
    x, y, w, h = cv2.boundingRect(np.array(landmarks))
    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    avg_color = np.mean(roi, axis=(0, 1))

    if avg_color[2] > 140:
        return "cool_skin"
    elif avg_color[2] < 120:
        return "warm_skin"
    else:
        return "neutral_skin"

# Function to detect hair color
def get_hair_color(image, face_rect):
    x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
    hair_region = image[max(0, y-50):y, x:x+w]  # Above forehead
    avg_color = np.mean(hair_region, axis=(0, 1))

    avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    if avg_color_hsv[0] < 20:
        return "black_hair"
    elif avg_color_hsv[0] < 30:
        return "brown_hair"
    elif avg_color_hsv[0] < 50:
        return "blonde_hair"
    else:
        return "red_hair"

# Function to determine face shape
def get_face_shape(landmarks):
    jaw_width = landmarks[16][0] - landmarks[0][0]
    face_height = landmarks[8][1] - landmarks[19][1]

    if jaw_width > face_height * 0.9:
        return "round_face"
    elif jaw_width < face_height * 0.8:
        return "oval_face"
    else:
        return "angular_face"

# Function to suggest dress colors
def suggest_dress_color(skin_tone, hair_color, face_shape):
    color_recommendations = {
        "warm_skin": ["earthy tones", "olive green", "warm reds"],
        "cool_skin": ["cool blues", "emerald green", "purple"],
        "neutral_skin": ["most colors", "soft pastels", "deep jewel tones"],
        "black_hair": ["royal blue", "emerald green"],
        "brown_hair": ["warm neutrals", "earthy tones"],
        "blonde_hair": ["pastel shades", "soft blue"],
        "red_hair": ["forest green", "warm brown"],
        "round_face": ["darker colors", "vertical patterns"],
        "angular_face": ["softer colors", "pastels"],
        "oval_face": ["most colors suit well"]
    }

    dress_colors = set()
    dress_colors.update(color_recommendations.get(skin_tone, []))
    dress_colors.update(color_recommendations.get(hair_color, []))
    dress_colors.update(color_recommendations.get(face_shape, []))

    return list(dress_colors)

# Image Processing Route
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return jsonify({"error": "No face detected."})

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        skin_tone = get_skin_tone(image, landmarks)
        hair_color = get_hair_color(image, face)
        face_shape = get_face_shape(landmarks)

        colors = suggest_dress_color(skin_tone, hair_color, face_shape)

        return jsonify({
            "Skin Tone": skin_tone,
            "Hair Color": hair_color,
            "Face Shape": face_shape,
            "Recommended Dress Colors": colors
        })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
