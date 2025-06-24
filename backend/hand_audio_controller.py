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
from collections import deque

class RealTimeAudioProcessor:
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size
        self.audio_buffer = deque()
        self.processed_buffer = deque()
        self.current_position = 0
        self.is_playing = False
        self.pitch_factor = 1.0
        self.tempo_factor = 1.0
        self.original_audio = None
        self.original_sr = None
        self.lock = threading.Lock()
        
    def load_audio(self, audio_data, sr):
        """Load audio data for real-time processing"""
        with self.lock:
            self.original_audio = audio_data
            self.original_sr = sr
            self.current_position = 0
            self.audio_buffer.clear()
            self.processed_buffer.clear()
            
            # Split audio into chunks
            for i in range(0, len(audio_data), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    # Pad the last chunk
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                self.audio_buffer.append(chunk)
    
    def update_parameters(self, pitch_factor, tempo_factor):
        """Update audio parameters for real-time modification"""
        with self.lock:
            self.pitch_factor = pitch_factor
            self.tempo_factor = tempo_factor
    
    def get_next_chunk(self):
        """Get the next processed audio chunk"""
        with self.lock:
            if not self.audio_buffer or self.current_position >= len(self.audio_buffer):
                # Loop back to beginning
                self.current_position = 0
                
            if self.current_position < len(self.audio_buffer):
                chunk = self.audio_buffer[self.current_position]
                
                # Apply real-time effects
                processed_chunk = self.apply_effects(chunk)
                
                self.current_position += 1
                return processed_chunk
            
            return np.zeros(self.chunk_size)
    
    def apply_effects(self, chunk):
        """Apply pitch and tempo effects to a chunk with overlap-add processing"""
        try:
            # Apply pitch shift with overlap-add
            if abs(self.pitch_factor - 1.0) > 0.01:
                semitones = 12 * np.log2(self.pitch_factor)
                # Use a larger window for pitch shift
                window_size = min(8192, len(chunk))
                hop_size = window_size // 4
                
                # Split chunk into overlapping windows
                windows = []
                for i in range(0, len(chunk) - window_size + 1, hop_size):
                    window = chunk[i:i + window_size]
                    window = librosa.effects.pitch_shift(window, sr=self.original_sr, n_steps=semitones)
                    windows.append(window)
                
                # Reconstruct using overlap-add
                chunk = np.zeros(len(chunk))
                for i, window in enumerate(windows):
                    start_idx = i * hop_size
                    chunk[start_idx:start_idx + window_size] += window
                
                # Normalize to prevent clipping
                chunk /= np.max(np.abs(chunk))
            
            # Apply tempo change with more accurate time stretching
            if abs(self.tempo_factor - 1.0) > 0.01:
                # Use phase vocoder for better quality
                chunk = librosa.phase_vocoder(librosa.stft(chunk), 
                                            rate=self.tempo_factor, 
                                            hop_length=512)
                chunk = librosa.istft(chunk)
                
                # Adjust chunk size after time stretching
                if len(chunk) > self.chunk_size:
                    chunk = chunk[:self.chunk_size]
                elif len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
            
            return chunk
        except Exception as e:
            print(f"Effect processing error: {e}")
            return chunk

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
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        
        # Real-time audio processor
        self.audio_processor = RealTimeAudioProcessor(chunk_size=512)
        
        # Audio parameters
        self.original_audio = None
        self.original_sr = None
        self.audio_playing = False
        self.audio_thread = None
        
        # Hand tracking parameters
        self.left_hand_distance = 0.1
        self.right_hand_distance = 0.1
        self.pitch_factor = 1.0
        self.tempo_factor = 1.0
        
        # Smoothing for parameters
        self.pitch_history = deque(maxlen=5)
        self.tempo_history = deque(maxlen=5)
        
        # Distance ranges for mapping
        self.min_distance = 0.02
        self.max_distance = 0.15
        self.pitch_range = (0.5, 2.0)
        self.tempo_range = (0.5, 2.0)
        
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
        distance = max(self.min_distance, min(self.max_distance, distance))
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        return factor_range[0] + normalized * (factor_range[1] - factor_range[0])
    
    def smooth_parameter(self, new_value, history):
        """Apply smoothing to parameters for stable audio"""
        history.append(new_value)
        return sum(history) / len(history)
    
    def process_hands(self, frame):
        """Process hand landmarks and update audio parameters"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                distance = self.calculate_distance(thumb_tip, index_tip)
                
                if hand_label == "Left":
                    self.left_hand_distance = distance
                    raw_pitch_factor = self.map_distance_to_factor(distance, self.pitch_range)
                    self.pitch_factor = self.smooth_parameter(raw_pitch_factor, self.pitch_history)
                elif hand_label == "Right":
                    self.right_hand_distance = distance
                    raw_tempo_factor = self.map_distance_to_factor(distance, self.tempo_range)
                    self.tempo_factor = self.smooth_parameter(raw_tempo_factor, self.tempo_history)
                
                # Update audio processor parameters in real-time
                self.audio_processor.update_parameters(self.pitch_factor, self.tempo_factor)
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw distance line
                h, w, _ = frame.shape
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 2)
                
                text = f"{hand_label}: {distance:.3f}"
                cv2.putText(frame, text, (thumb_pos[0], thumb_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def audio_playback_loop(self):
        """Continuous audio playback loop with real-time modifications"""
        print("Starting audio playback loop...")
        
        while self.audio_playing:
            try:
                # Get next processed chunk
                chunk = self.audio_processor.get_next_chunk()
                
                if chunk is not None and len(chunk) > 0:
                    # Convert to pygame format
                    chunk = np.clip(chunk, -1.0, 1.0)
                    chunk = (chunk * 32767).astype(np.int16)
                    
                    # Create and play sound
                    try:
                        sound = pygame.sndarray.make_sound(chunk)
                        channel = pygame.mixer.find_channel()
                        if channel:
                            channel.play(sound)
                        else:
                            # If no free channel, wait a bit
                            time.sleep(0.01)
                    except Exception as e:
                        print(f"Pygame sound error: {e}")
                        time.sleep(0.01)
                
                # Control playback rate
                time.sleep(0.02)  # ~50 FPS audio updates
                
            except Exception as e:
                print(f"Audio playback error: {e}")
                time.sleep(0.1)
        
        print("Audio playback loop stopped")
    
    def load_audio(self, audio_file):
        """Load audio file"""
        try:
            self.original_audio, self.original_sr = librosa.load(audio_file, sr=22050)
            self.audio_processor.load_audio(self.original_audio, self.original_sr)
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def start_audio(self):
        """Start continuous audio playback"""
        if self.original_audio is None:
            return False
        
        if self.audio_playing:
            return True
        
        self.audio_playing = True
        self.audio_thread = threading.Thread(target=self.audio_playback_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        return True
    
    def stop_audio(self):
        """Stop audio playback"""
        self.audio_playing = False
        pygame.mixer.stop()
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
    
    def update_visualization(self):
        """Update real-time visualization"""
        current_time = time.time()
        
        try:
            self.time_data.put_nowait(current_time)
            self.pitch_data.put_nowait(self.pitch_factor)
            self.tempo_data.put_nowait(self.tempo_factor)
        except queue.Full:
            try:
                self.time_data.get_nowait()
                self.pitch_data.get_nowait() 
                self.tempo_data.get_nowait()
                self.time_data.put_nowait(current_time)
                self.pitch_data.put_nowait(self.pitch_factor)
                self.tempo_data.put_nowait(self.tempo_factor)
            except queue.Empty:
                pass
        
        times = list(self.time_data.queue)
        pitches = list(self.pitch_data.queue)
        tempos = list(self.tempo_data.queue)
        
        if len(times) > 1:
            times = np.array(times) - times[0] if times else []
            
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            self.ax1.plot(times, pitches, 'b-', linewidth=2, label='Pitch Factor')
            self.ax1.set_ylabel('Pitch Factor')
            self.ax1.set_title('Real-time Audio Modifications')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            self.ax1.set_ylim(self.pitch_range[0] - 0.1, self.pitch_range[1] + 0.1)
            
            self.ax2.plot(times, tempos, 'r-', linewidth=2, label='Tempo Factor')
            self.ax2.set_ylabel('Tempo Factor')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            self.ax2.set_ylim(self.tempo_range[0] - 0.1, self.tempo_range[1] + 0.1)
            
            left_distances = [self.left_hand_distance] * len(times)
            right_distances = [self.right_hand_distance] * len(times)
            
            self.ax3.plot(times, left_distances, 'g-', linewidth=2, label='Left Hand Distance')
            self.ax3.plot(times, right_distances, 'm-', linewidth=2, label='Right Hand Distance')
            self.ax3.set_xlabel('Time (seconds)')
            self.ax3.set_ylabel('Hand Distance')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.legend()
            self.ax3.set_ylim(0, self.max_distance + 0.02)
        
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
        
        processed_frame = self.process_hands(frame)
        
        info_text = [
            f"Pitch Factor: {self.pitch_factor:.2f} (Left Hand)",
            f"Tempo Factor: {self.tempo_factor:.2f} (Right Hand)",
            f"Left Distance: {self.left_hand_distance:.3f}",
            f"Right Distance: {self.right_hand_distance:.3f}",
            f"Audio Playing: {'Yes' if self.audio_playing else 'No'}"
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
        
        temp_path = f"temp_audio_{int(time.time())}.wav"
        audio_file.save(temp_path)
        
        success = controller.load_audio(temp_path)
        
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
        
        success = controller.start_audio()
        if success:
            return jsonify({'message': 'Audio playback started'})
        else:
            return jsonify({'error': 'Failed to start audio'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_audio', methods=['POST'])
def stop_audio():
    """Stop audio playback"""
    try:
        controller.stop_audio()
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
    print("Starting Real-time Hand Gesture Audio Controller Backend...")
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