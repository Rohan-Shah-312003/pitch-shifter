import cv2
import mediapipe as mp
import numpy as np
import soundfile as sf
import librosa
import pygame
import threading
import time
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import io
import base64
from scipy.signal import resample
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import queue

class HandAudioController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        # Audio parameters
        self.original_audio = None
        self.original_sr = None
        self.current_audio = None
        self.audio_playing = False
        
        # Hand tracking parameters
        self.left_hand_distance = 0.1  # Default distance (normalized)
        self.right_hand_distance = 0.1
        self.pitch_factor = 1.0  # No pitch change initially
        self.tempo_factor = 1.0  # No tempo change initially
        
        # Distance ranges for mapping
        self.min_distance = 0.02  # Minimum finger distance
        self.max_distance = 0.15  # Maximum finger distance
        self.pitch_range = (0.5, 2.0)  # Pitch multiplier range
        self.tempo_range = (0.5, 2.0)  # Tempo multiplier range
        
        # Visualization
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Data queues for real-time plotting
        self.time_data = queue.Queue(maxsize=100)
        self.pitch_data = queue.Queue(maxsize=100)
        self.tempo_data = queue.Queue(maxsize=100)
        
        # Camera
        self.cap = None
        self.camera_running = False
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def map_distance_to_factor(self, distance, factor_range):
        """Map normalized distance to a factor within the given range"""
        # Clamp distance to valid range
        distance = max(self.min_distance, min(self.max_distance, distance))
        
        # Normalize to 0-1
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        
        # Map to factor range
        return factor_range[0] + normalized * (factor_range[1] - factor_range[0])
    
    def process_hands(self, frame):
        """Process hand landmarks and update audio parameters"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                # Get thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Calculate distance
                distance = self.calculate_distance(thumb_tip, index_tip)
                
                # Update parameters based on hand
                if hand_label == "Left":
                    self.left_hand_distance = distance
                    self.pitch_factor = self.map_distance_to_factor(distance, self.pitch_range)
                elif hand_label == "Right":
                    self.right_hand_distance = distance
                    self.tempo_factor = self.map_distance_to_factor(distance, self.tempo_range)
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw distance line
                h, w, _ = frame.shape
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 2)
                
                # Add text with current values
                text = f"{hand_label}: {distance:.3f}"
                cv2.putText(frame, text, (thumb_pos[0], thumb_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def modify_audio(self, audio_data, sr):
        """Apply pitch and tempo modifications to audio"""
        try:
            # Apply pitch shift
            if self.pitch_factor != 1.0:
                # Convert pitch factor to semitones
                semitones = 12 * np.log2(self.pitch_factor)
                audio_data = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=semitones)
            
            # Apply tempo change
            if self.tempo_factor != 1.0:
                audio_data = librosa.effects.time_stretch(audio_data, rate=self.tempo_factor)
            
            return audio_data
        except Exception as e:
            print(f"Audio modification error: {e}")
            return audio_data
    
    def load_audio(self, audio_file):
        """Load audio file"""
        try:
            self.original_audio, self.original_sr = librosa.load(audio_file, sr=None)
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def play_audio(self):
        """Play modified audio in a loop"""
        if self.original_audio is None:
            return
        
        while self.audio_playing:
            try:
                # Apply current modifications
                modified_audio = self.modify_audio(self.original_audio.copy(), self.original_sr)
                
                # Convert to pygame format
                modified_audio = np.clip(modified_audio, -1.0, 1.0)
                modified_audio = (modified_audio * 32767).astype(np.int16)
                
                # Create pygame sound
                sound_array = np.array([modified_audio, modified_audio]).T
                sound = pygame.sndarray.make_sound(sound_array)
                
                # Play sound
                pygame.mixer.stop()
                pygame.mixer.Sound.play(sound)
                
                # Wait for sound to finish or parameters to change significantly
                duration = len(modified_audio) / self.original_sr
                sleep_time = max(0.1, duration / 10)  # Update frequently for responsiveness
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Audio playback error: {e}")
                time.sleep(0.1)
    
    def update_visualization(self):
        """Update real-time visualization"""
        current_time = time.time()
        
        # Add current data to queues
        try:
            self.time_data.put_nowait(current_time)
            self.pitch_data.put_nowait(self.pitch_factor)
            self.tempo_data.put_nowait(self.tempo_factor)
        except queue.Full:
            # Remove oldest data if queue is full
            try:
                self.time_data.get_nowait()
                self.pitch_data.get_nowait() 
                self.tempo_data.get_nowait()
                self.time_data.put_nowait(current_time)
                self.pitch_data.put_nowait(self.pitch_factor)
                self.tempo_data.put_nowait(self.tempo_factor)
            except queue.Empty:
                pass
        
        # Convert queues to lists for plotting
        times = list(self.time_data.queue)
        pitches = list(self.pitch_data.queue)
        tempos = list(self.tempo_data.queue)
        
        if len(times) > 1:
            # Normalize time to start from 0
            times = np.array(times) - times[0] if times else []
            
            # Clear and update plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Plot pitch factor
            self.ax1.plot(times, pitches, 'b-', linewidth=2, label='Pitch Factor')
            self.ax1.set_ylabel('Pitch Factor')
            self.ax1.set_title('Real-time Audio Modifications')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            self.ax1.set_ylim(self.pitch_range[0] - 0.1, self.pitch_range[1] + 0.1)
            
            # Plot tempo factor
            self.ax2.plot(times, tempos, 'r-', linewidth=2, label='Tempo Factor')
            self.ax2.set_ylabel('Tempo Factor')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            self.ax2.set_ylim(self.tempo_range[0] - 0.1, self.tempo_range[1] + 0.1)
            
            # Plot hand distances
            left_distances = [self.left_hand_distance] * len(times)
            right_distances = [self.right_hand_distance] * len(times)
            
            self.ax3.plot(times, left_distances, 'g-', linewidth=2, label='Left Hand Distance')
            self.ax3.plot(times, right_distances, 'm-', linewidth=2, label='Right Hand Distance')
            self.ax3.set_xlabel('Time (seconds)')
            self.ax3.set_ylabel('Hand Distance')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.legend()
            self.ax3.set_ylim(0, self.max_distance + 0.02)
        
        # Save plot as base64 image
        canvas = FigureCanvasAgg(self.fig)
        png_output = io.BytesIO()
        canvas.print_png(png_output)
        png_output.seek(0)
        plot_data = base64.b64encode(png_output.read()).decode()
        return plot_data
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        self.camera_running = True
        return self.cap.isOpened()
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        """Get processed camera frame"""
        if not self.cap or not self.camera_running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Process hands and get modified frame
        processed_frame = self.process_hands(frame)
        
        # Add info text
        info_text = [
            f"Pitch Factor: {self.pitch_factor:.2f} (Left Hand)",
            f"Tempo Factor: {self.tempo_factor:.2f} (Right Hand)",
            f"Left Distance: {self.left_hand_distance:.3f}",
            f"Right Distance: {self.right_hand_distance:.3f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(processed_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return processed_frame

# Flask app for API
app = Flask(__name__)
CORS(app)

# Global controller instance
controller = HandAudioController()

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Upload and load audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save temporarily and load
        temp_path = f"temp_audio_{int(time.time())}.wav"
        audio_file.save(temp_path)
        
        success = controller.load_audio(temp_path)
        
        # Clean up temp file
        import os
        try:
            os.remove(temp_path)
        except:
            pass
        
        if success:
            return jsonify({'message': 'Audio loaded successfully'})
        else:
            return jsonify({'error': 'Failed to load audio'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_audio', methods=['POST'])
def start_audio():
    """Start audio playback"""
    try:
        if controller.original_audio is None:
            return jsonify({'error': 'No audio loaded'}), 400
        
        controller.audio_playing = True
        audio_thread = threading.Thread(target=controller.play_audio)
        audio_thread.daemon = True
        audio_thread.start()
        
        return jsonify({'message': 'Audio playback started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_audio', methods=['POST'])
def stop_audio():
    """Stop audio playback"""
    try:
        controller.audio_playing = False
        pygame.mixer.stop()
        return jsonify({'message': 'Audio playback stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera capture"""
    try:
        success = controller.start_camera()
        if success:
            return jsonify({'message': 'Camera started successfully'})
        else:
            return jsonify({'error': 'Failed to start camera'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    try:
        controller.stop_camera()
        return jsonify({'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_frames():
    """Generate camera frames for streaming"""
    while controller.camera_running:
        frame = controller.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data', methods=['GET'])
def get_data():
    """Get current hand tracking and audio data"""
    try:
        data = {
            'left_hand_distance': controller.left_hand_distance,
            'right_hand_distance': controller.right_hand_distance,
            'pitch_factor': controller.pitch_factor,
            'tempo_factor': controller.tempo_factor,
            'audio_playing': controller.audio_playing,
            'camera_running': controller.camera_running
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_visualization', methods=['GET'])
def get_visualization():
    """Get current visualization plot as base64 image"""
    try:
        plot_data = controller.update_visualization()
        return jsonify({'plot': plot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Hand Gesture Audio Controller Backend...")
    print("Available endpoints:")
    print("- POST /upload_audio - Upload audio file")
    print("- POST /start_audio - Start audio playback")
    print("- POST /stop_audio - Stop audio playback") 
    print("- POST /start_camera - Start camera")
    print("- POST /stop_camera - Stop camera")
    print("- GET /video_feed - Camera stream")
    print("- GET /get_data - Get current data")
    print("- GET /get_visualization - Get visualization plot")
    print("- GET /health - Health check")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)