import cv2
import mediapipe as mp
import math
import librosa
import sounddevice as sd
import pyrubberband as pyrb
import numpy as np
import time
import base64
from queue import Queue, Empty
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global state
is_running = False
pitch_value = 0
rate_value = 1.0
frames_queue = Queue(maxsize=5)
pitch_queue = Queue(maxsize=1)
rate_queue = Queue(maxsize=1)
status_message = "🤚 No hands detected"
current_fps = 0

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Audio setup
y_full, sr = None, None
chunk_size = 2048
play_ptr = 0
stream = None
audio_thread = None

# Initialize hands detection
hands = mp_hands.Hands(
    model_complexity=0,
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_audio():
    global y_full, sr
    print("Loading audio sample...")
    y_full, sr = librosa.load(librosa.example('brahms'), mono=True)
    print("Audio loaded")

def audio_callback(outdata, frames, time_info, status):
    global play_ptr
    chunk = y_full[play_ptr:play_ptr+chunk_size]

    if len(chunk) < chunk_size:
        play_ptr = 0
        chunk = y_full[play_ptr:play_ptr+chunk_size]
    play_ptr += chunk_size

    try:
        pitch = pitch_queue.get_nowait()
    except Empty:
        pitch = 0

    try:
        rate = rate_queue.get_nowait()
    except Empty:
        rate = 1.0

    try:
        modified = pyrb.pitch_shift(chunk, sr, n_steps=pitch)
        modified = pyrb.time_stretch(modified, sr, rate)

        if len(modified) < chunk_size:
            modified = np.pad(modified, (0, chunk_size - len(modified)))
        else:
            modified = modified[:chunk_size]

        outdata[:] = modified.reshape(-1, 1)

    except Exception as e:
        print(f"[Audio error] {e}")
        outdata[:] = chunk[:chunk_size].reshape(-1, 1)

def start_audio_stream():
    global stream
    if stream is None:
        stream = sd.OutputStream(
            samplerate=sr,
            blocksize=chunk_size,
            channels=1,
            callback=audio_callback
        )
        stream.start()
        print("Audio stream started")

def stop_audio_stream():
    global stream
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
        print("Audio stream stopped")

def process_frame(frame):
    global pitch_value, rate_value, status_message, current_fps
    
    if frame is None:
        return None
    
    # Convert frame to RGB for MediaPipe
    frame = cv2.flip(frame, 1)  # Mirror
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(frame_rgb)
    
    # Create a copy for drawing
    overlay = frame.copy()
    status_message = "🤚 No hands detected"
    colors = [(0, 255, 0), (255, 0, 0)]  # Green, Blue
    
    if results.multi_hand_landmarks:
        status_message = "🎵 Adjusting Pitch & Tempo"
        
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks, start=1):
            mp_drawing.draw_landmarks(
                overlay,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(idx_tip.x * w), int(idx_tip.y * h)
            
            dist = math.hypot(x2 - x1, y2 - y1) / 100
            
            if hand_no == 1:
                pitch_value = float(np.interp(dist, [1, 5], [-12, 12]))
                pitch_queue.queue.clear()
                pitch_queue.put(pitch_value)
            else:
                rate_value = float(np.interp(dist, [1, 5], [0.5, 2.0]))
                rate_queue.queue.clear()
                rate_queue.put(rate_value)
            
            # Draw finger points
            cv2.circle(overlay, (x1, y1), 10, colors[hand_no - 1], -1)
            cv2.circle(overlay, (x2, y2), 10, colors[hand_no - 1], -1)
    
    # Add transparent UI background
    cv2.rectangle(overlay, (0, 0), (350, 160), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Add text overlays
    cv2.putText(result, "🎧 Real-Time Audio Shifter", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, status_message, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Draw bars
    draw_bar(result, 10, 90, pitch_value, -12, 12, (0, 255, 0), "Pitch")
    draw_bar(result, 10, 130, rate_value, 0.5, 2.0, (255, 0, 0), "Tempo")
    
    # FPS
    cv2.putText(result, f"FPS: {int(current_fps)}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return result

def draw_bar(image, x, y, value, min_val, max_val, color, label):
    bar_length = 200
    norm_val = np.clip((value - min_val) / (max_val - min_val), 0, 1)
    filled_length = int(norm_val * bar_length)

    cv2.rectangle(image, (x, y), (x + bar_length, y + 20), (40, 40, 40), -1)
    cv2.rectangle(image, (x, y), (x + filled_length, y + 20), color, -1)
    cv2.putText(image, f"{label}: {value:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def video_processing_thread():
    global is_running, current_fps, frames_queue
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    prev_time = time.time()
    
    try:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            processed_frame = process_frame(frame)
            
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Put in queue, overwrite old frames if full
            if frames_queue.full():
                try:
                    frames_queue.get_nowait()
                except Empty:
                    pass
            frames_queue.put(jpg_as_text)
            
            # Calculate FPS
            curr_time = time.time()
            current_fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Don't run too fast
            time.sleep(0.01)
    finally:
        cap.release()
        print("Video capture released")

@app.route('/api/start', methods=['POST'])
def start_processing():
    global is_running, audio_thread
    
    if is_running:
        return jsonify({"status": "already_running"})
    
    # Load audio if not loaded
    if y_full is None:
        load_audio()
    
    # Start audio stream
    start_audio_stream()
    
    # Start video processing thread
    is_running = True
    audio_thread = threading.Thread(target=video_processing_thread)
    audio_thread.daemon = True
    audio_thread.start()
    
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    global is_running
    
    if not is_running:
        return jsonify({"status": "not_running"})
    
    # Stop video processing
    is_running = False
    
    # Stop audio
    stop_audio_stream()
    
    # Clear frame queue
    while not frames_queue.empty():
        try:
            frames_queue.get_nowait()
        except Empty:
            pass
    
    return jsonify({"status": "stopped"})

@app.route('/api/frame')
def get_frame():
    if not is_running:
        return jsonify({"error": "Not running"}), 400
    
    try:
        # Non-blocking get with timeout
        frame_data = frames_queue.get(timeout=0.5)
        return jsonify({
            "frame": frame_data,
            "status": status_message,
            "pitch": pitch_value,
            "rate": rate_value,
            "fps": int(current_fps)
        })
    except Empty:
        return jsonify({"error": "No frames available"}), 404

@app.route('/api/status')
def get_status():
    return jsonify({
        "running": is_running,
        "status": status_message,
        "pitch": pitch_value,
        "rate": rate_value,
        "fps": int(current_fps)
    })

@app.route('/')
def index():
    return "Hand Audio Controller API is running. Connect with your Next.js frontend."

if __name__ == '__main__':
    print("Starting Flask server...")
    # Load audio sample at startup
    load_audio()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)