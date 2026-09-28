from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, timezone
import base64
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import os
from sqlalchemy import desc, func

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/checkpoint.pth"
CLASS_NAMES = [
    # Basic Greetings
    "Hello", "Goodbye", "ThankYou", "Please",
    # Common Responses
    "Yes", "No", "Maybe", "OK",
    # Basic Needs
    "Help", "Water", "Food", "Bathroom",
    # Emotions
    "Happy", "Sad", "Love", "Sorry",
    # Time Related
    "Today", "Tomorrow", "Later", "Now",
    # Questions
    "What", "Where", "When", "How",
    # Numbers
    "One", "Two", "Three", "Four", "Five",
    # Common Actions
    "Want", "Need", "Give", "Take",
    # Family
    "Mother", "Father", "Family", "Friend"
]

# -------------------- FLASK APP --------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sign_language.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------- DATABASE MODELS --------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    landmarks_detected = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.String(500), nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    session_id = db.Column(db.String(50), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'confidence': self.confidence,
            'landmarks_detected': self.landmarks_detected,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id
        }

class GestureInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    image_path = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class Statistics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_predictions = db.Column(db.Integer, default=0)
    successful_predictions = db.Column(db.Integer, default=0)
    failed_predictions = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# -------------------- DEFINE MODEL --------------------
class LandmarkTransformer(nn.Module):
    def __init__(self, input_dim=63, embed_dim=64, num_heads=4, num_layers=2, num_classes=36):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.fc(x)
        return x

# -------------------- LOAD MODEL --------------------
model = LandmarkTransformer(input_dim=63, num_classes=len(CLASS_NAMES)).to(DEVICE)
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# -------------------- MEDIAPIPE SETUP --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -------------------- HELPER FUNCTIONS --------------------
def update_statistics(success=True):
    stats = Statistics.query.first()
    if not stats:
        stats = Statistics(total_predictions=0, successful_predictions=0, failed_predictions=0)
        db.session.add(stats)
        db.session.commit()
    
    stats.total_predictions += 1
    if success:
        stats.successful_predictions += 1
    else:
        stats.failed_predictions += 1
    stats.last_updated = datetime.now(timezone.utc)
    db.session.commit()

def initialize_gestures():
    """Initialize gesture information in database"""
    categories = {
        "Hello": "Basic Greetings", "Goodbye": "Basic Greetings", 
        "ThankYou": "Basic Greetings", "Please": "Basic Greetings",
        "Yes": "Common Responses", "No": "Common Responses", 
        "Maybe": "Common Responses", "OK": "Common Responses",
        "Help": "Basic Needs", "Water": "Basic Needs", 
        "Food": "Basic Needs", "Bathroom": "Basic Needs",
        "Happy": "Emotions", "Sad": "Emotions", 
        "Love": "Emotions", "Sorry": "Emotions",
        "Today": "Time Related", "Tomorrow": "Time Related", 
        "Later": "Time Related", "Now": "Time Related",
        "What": "Questions", "Where": "Questions", 
        "When": "Questions", "How": "Questions",
        "One": "Numbers", "Two": "Numbers", "Three": "Numbers", "Four": "Numbers", "Five": "Numbers",
        "Want": "Common Actions", "Need": "Common Actions", "Give": "Common Actions", "Take": "Common Actions",
        "Mother": "Family", "Father": "Family", "Family": "Family", "Friend": "Family"
    }
    for name, category in categories.items():
        if not GestureInfo.query.filter_by(name=name).first():
            gesture = GestureInfo(name=name, category=category, description=f"Sign for {name.lower()} in sign language.")
            db.session.add(gesture)
    db.session.commit()

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/gestures")
def gestures():
    category = request.args.get('category', None)
    if category:
        gesture_list = GestureInfo.query.filter_by(category=category).order_by(GestureInfo.name).all()
    else:
        gesture_list = GestureInfo.query.order_by(GestureInfo.name).all()
    
    categories = db.session.query(GestureInfo.category).distinct().all()
    categories = [cat[0] for cat in categories]
    
    return render_template("gestures.html", gestures=gesture_list, categories=categories, selected_category=category)

@app.route("/history")
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    predictions = Prediction.query.order_by(desc(Prediction.timestamp)).paginate(page=page, per_page=per_page, error_out=False)
    return render_template("history.html", predictions=predictions)

@app.route("/statistics")
def statistics():
    stats = Statistics.query.first()
    if not stats:
        stats = Statistics(total_predictions=0, successful_predictions=0, failed_predictions=0)
        db.session.add(stats)
        db.session.commit()
    
    # Get predictions per gesture - Fixed: Convert Row to tuple for JSON serialization
    raw_gesture_stats = db.session.query(
        Prediction.label, 
        func.count(Prediction.id).label('count')
    ).filter(Prediction.landmarks_detected == True).group_by(Prediction.label).order_by(desc(func.count(Prediction.id))).limit(10).all()
    gesture_stats = [(stat.label, stat.count) for stat in raw_gesture_stats]
    
    # Get recent activity (last 7 days)
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_predictions = Prediction.query.filter(Prediction.timestamp >= week_ago).count()
    
    return render_template("statistics.html", 
                         stats=stats, 
                         gesture_stats=gesture_stats,
                         recent_predictions=recent_predictions)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    session_id = data.get("session_id", "unknown")

    try:
        # Decode base64 image
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Convert PIL to numpy array for MediaPipe
        image_np = np.array(image)
        
        # Process with MediaPipe
        results = hands.process(image_np)
        
        if not results.multi_hand_landmarks:
            prediction = Prediction(
                label="No Hand Detected",
                confidence=0.0,
                landmarks_detected=False,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc)
            )
            db.session.add(prediction)
            db.session.commit()
            update_statistics(success=False)
            
            return jsonify({
                "error": "No hand detected",
                "label": "No Hand Detected",
                "confidence": 0.0
            })
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Convert to tensor
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(landmarks_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = CLASS_NAMES[pred.item()]
        confidence = conf.item()
        
        # Save to database
        prediction = Prediction(
            label=label,
            confidence=confidence,
            landmarks_detected=True,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(prediction)
        db.session.commit()
        update_statistics(success=True)
        
        return jsonify({
            "label": label,
            "confidence": confidence,
            "landmarks_detected": True
        })
        
    except Exception as e:
        prediction = Prediction(
            label="Error",
            confidence=0.0,
            landmarks_detected=False,
            error_message=str(e),
            session_id=session_id,
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(prediction)
        db.session.commit()
        update_statistics(success=False)
        
        return jsonify({"error": str(e)}), 500

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    limit = request.args.get('limit', 50, type=int)
    predictions = Prediction.query.order_by(desc(Prediction.timestamp)).limit(limit).all()
    return jsonify([p.to_dict() for p in predictions])

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    try:
        Prediction.query.delete()
        # Reset statistics
        stats = Statistics.query.first()
        if stats:
            stats.total_predictions = 0
            stats.successful_predictions = 0
            stats.failed_predictions = 0
            stats.last_updated = datetime.now(timezone.utc)
        db.session.commit()
        return jsonify({"success": True, "message": "History cleared"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

# -------------------- INITIALIZE --------------------
with app.app_context():
    db.create_all()
    initialize_gestures()
    print("✅ Database initialized!")

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)