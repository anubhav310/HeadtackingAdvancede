import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import time

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Global variables for head position and system state
head_position = {"x": 0.5, "y": 0.5}
calibration_data = {"x": 0.5, "y": 0.5}
is_calibrated = False
cursor_smoothing = 0.7  # Smoothing factor for cursor movement
sensitivity = 3.0  # Sensitivity multiplier for head movement

def generate_frames():
    global head_position
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get nose tip landmark (index 1)
            nose_tip = face_landmarks.landmark[1]
            h, w, c = frame.shape
            
            # Calculate nose position
            nose_x = int(nose_tip.x * w)
            nose_y = int(nose_tip.y * h)
            
            # Draw a circle on the nose tip
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
            
            # Update head position
            head_position["x"] = nose_tip.x
            head_position["y"] = nose_tip.y
            
        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/head_position')
def get_head_position():
    global head_position, calibration_data, is_calibrated, sensitivity
    
    if is_calibrated:
        # Apply calibration with sensitivity
        calibrated_x = 0.5 + (head_position["x"] - calibration_data["x"]) * sensitivity
        calibrated_y = 0.5 + (head_position["y"] - calibration_data["y"]) * sensitivity
        
        # Constrain to screen bounds (0 to 1)
        calibrated_x = max(0.0, min(1.0, calibrated_x))
        calibrated_y = max(0.0, min(1.0, calibrated_y))
        
        return jsonify({
            "x": calibrated_x, 
            "y": calibrated_y,
            "calibrated": True
        })
    
    return jsonify({
        "x": head_position["x"], 
        "y": head_position["y"],
        "calibrated": False
    })

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global calibration_data, is_calibrated, head_position
    
    calibration_data["x"] = head_position["x"]
    calibration_data["y"] = head_position["y"]
    is_calibrated = True
    
    print(f"Calibration completed: {calibration_data}")
    
    return jsonify({"status": "success"})

@app.route('/reset_calibration', methods=['POST'])
def reset_calibration():
    global is_calibrated
    
    is_calibrated = False
    print("Calibration reset")
    
    return jsonify({"status": "success"})

@app.route('/adjust_sensitivity', methods=['POST'])
def adjust_sensitivity():
    global sensitivity
    data = request.json
    sensitivity = max(1.0, min(10.0, float(data.get('sensitivity', 3.0))))
    
    return jsonify({"status": "success", "sensitivity": sensitivity})

@app.route('/activate', methods=['POST'])
def activate_tile():
    data = request.json
    tile_id = data.get('tile')
    
    tile_names = {
        'water': 'Water',
        'food': 'Food',
        'help': 'Help',
        'comfort': 'Comfort',
        'toilet': 'Toilet',
        'pain': 'Pain Relief'
    }
    
    print(f"Tile activated: {tile_names.get(tile_id, tile_id)} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return jsonify({"status": "success", "tile": tile_id})

@app.route('/emergency', methods=['POST'])
def emergency():
    print(f"EMERGENCY activated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return jsonify({"status": "emergency_activated"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')