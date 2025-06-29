import cv2
import mediapipe as mp
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import pygame
import threading
import time
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
    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.overlap_size = chunk_size // 8  # 25% overlap for smoother transitions
        self.audio_buffer = deque()
        self.processed_buffer = deque()
        self.overlap_buffer = np.zeros(self.overlap_size)
        self.current_position = 0
        self.is_playing = False
        self.pitch_factor = 1.0
        self.tempo_factor = 1.0
        self.original_audio = None
        self.original_sr = None
        self.lock = threading.Lock()
        
        # Smoothing parameters for pyrubberband
        self.pitch_smoothing = 0.1  # Lower = more responsive
        self.tempo_smoothing = 0.1
        self.smoothed_pitch = 1.0
        self.smoothed_tempo = 1.0
        
    def load_audio(self, audio_data, sr):
        """Load audio data for real-time processing"""
        with self.lock:
            self.original_audio = audio_data
            self.original_sr = sr
            self.current_position = 0
            self.audio_buffer.clear()
            self.processed_buffer.clear()
            self.overlap_buffer = np.zeros(self.overlap_size)
            
            # Split audio into overlapping chunks for smoother processing
            for i in range(0, len(audio_data), self.chunk_size - self.overlap_size):
                chunk = audio_data[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    # Pad the last chunk
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                self.audio_buffer.append(chunk)
    
    def update_parameters(self, pitch_factor, tempo_factor):
        """Update audio parameters with smoothing for real-time modification"""
        with self.lock:
            # Apply exponential smoothing for stable parameter changes
            self.smoothed_pitch = (self.pitch_smoothing * pitch_factor + 
                                 (1 - self.pitch_smoothing) * self.smoothed_pitch)
            self.smoothed_tempo = (self.tempo_smoothing * tempo_factor + 
                                 (1 - self.tempo_smoothing) * self.smoothed_tempo)
            
            self.pitch_factor = self.smoothed_pitch
            self.tempo_factor = self.smoothed_tempo
    
    def get_next_chunk(self):
        """Get the next processed audio chunk with overlap-add"""
        with self.lock:
            if not self.audio_buffer or self.current_position >= len(self.audio_buffer):
                # Loop back to beginning
                self.current_position = 0
                self.overlap_buffer = np.zeros(self.overlap_size)
                
            if self.current_position < len(self.audio_buffer):
                chunk = self.audio_buffer[self.current_position].copy()
                
                # Apply real-time effects using pyrubberband
                processed_chunk = self.apply_effects(chunk)
                
                # Apply overlap-add for smooth transitions
                if self.current_position > 0:
                    # Blend with overlap from previous chunk
                    processed_chunk[:self.overlap_size] = (
                        processed_chunk[:self.overlap_size] * 0.5 + 
                        self.overlap_buffer * 0.5
                    )
                
                # Store overlap for next chunk
                self.overlap_buffer = processed_chunk[-self.overlap_size:].copy()
                
                self.current_position += 1
                return processed_chunk[:-self.overlap_size]  # Remove overlap portion
            
            return np.zeros(self.chunk_size - self.overlap_size)
    
    def apply_effects(self, chunk):
        """Apply pitch and tempo effects using pyrubberband"""
        try:
            # Ensure chunk is the right type and has sufficient length
            if len(chunk) < 512:  # Minimum length for pyrubberband
                return chunk
                
            chunk = chunk.astype(np.float64)
            
            # Apply pitch shift if needed (pyrubberband is much more efficient)
            if abs(self.pitch_factor - 1.0) > 0.01:
                # Convert pitch factor to semitones for pyrubberband
                semitones = 12 * np.log2(max(0.1, self.pitch_factor))
                try:
                    chunk = pyrb.pitch_shift(chunk, self.original_sr, semitones)
                except Exception as e:
                    print(f"Pitch shift error: {e}")
                    # Fallback: return original chunk if pyrubberband fails
                    pass
            
            # Apply tempo change if needed
            if abs(self.tempo_factor - 1.0) > 0.01:
                try:
                    # Pyrubberband time stretch (much faster than librosa)
                    stretch_factor = 1.0 / max(0.1, self.tempo_factor)
                    chunk = pyrb.time_stretch(chunk, self.original_sr, stretch_factor)
                    
                    # Adjust chunk size after time stretching
                    if len(chunk) > self.chunk_size:
                        chunk = chunk[:self.chunk_size]
                    elif len(chunk) < self.chunk_size:
                        chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                        
                except Exception as e:
                    print(f"Time stretch error: {e}")
                    # Fallback: return original chunk if pyrubberband fails
                    pass
            
            # Normalize to prevent clipping and ensure good dynamic range
            max_val = np.max(np.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val * 0.8  # Leave some headroom
            
            return chunk.astype(np.float32)
            
        except Exception as e:
            print(f"Effect processing error: {e}")
            return chunk.astype(np.float32)

class HandAudioController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize pygame for audio playback with better settings for real-time
        pygame.mixer.pre_init(
            frequency=22050, 
            size=-16, 
            channels=1, 
            buffer=256  # Smaller buffer for lower latency
        )
        pygame.mixer.init()
        pygame.mixer.set_num_channels(8)  # More channels for smoother playback
        
        # Real-time audio processor with optimized chunk size
        self.audio_processor = RealTimeAudioProcessor(chunk_size=1024)
        
        # Audio parameters
        self.original_audio = None
        self.original_sr = None
        self.audio_playing = False
        self.audio_thread = None
        
        # Hand tracking parameters with better smoothing
        self.left_hand_distance = 0.1
        self.right_hand_distance = 0.1
        self.pitch_factor = 1.0
        self.tempo_factor = 1.0
        
        # Enhanced smoothing for parameters
        self.pitch_history = deque(maxlen=8)  # More history for smoother control
        self.tempo_history = deque(maxlen=8)
        
        # Improved distance ranges for better control
        self.min_distance = 0.015  # Slightly smaller for more sensitivity
        self.max_distance = 0.18   # Slightly larger for more range
        self.pitch_range = (0.6, 1.8)  # More reasonable range
        self.tempo_range = (0.6, 1.8)
        
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
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def map_distance_to_factor(self, distance, factor_range):
        """Map normalized distance to a factor within the given range with smoothing"""
        distance = max(self.min_distance, min(self.max_distance, distance))
        # Use exponential mapping for more intuitive control
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        # Apply slight curve for better control feel
        normalized = normalized ** 0.8
        return factor_range[0] + normalized * (factor_range[1] - factor_range[0])
    
    def smooth_parameter(self, new_value, history):
        """Apply enhanced smoothing to parameters for stable audio"""
        history.append(new_value)
        # Use weighted average with more weight on recent values
        weights = np.linspace(0.5, 1.0, len(history))
        weights /= weights.sum()
        return np.average(list(history), weights=weights)
    
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
                
                # Draw hand landmarks with better visualization
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 255, 255), thickness=1
                    )
                )
                
                # Draw distance line with color coding
                h, w, _ = frame.shape
                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Color based on distance
                color = (0, 255, 0) if distance < 0.05 else (0, 255, 255) if distance < 0.1 else (0, 0, 255)
                cv2.line(frame, thumb_pos, index_pos, color, 3)
                
                # Enhanced text display
                factor = self.pitch_factor if hand_label == "Left" else self.tempo_factor
                param_name = "Pitch" if hand_label == "Left" else "Tempo"
                text = f"{hand_label} {param_name}: {factor:.2f}x"
                
                # Background rectangle for better text visibility
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, 
                            (thumb_pos[0] - 5, thumb_pos[1] - 35),
                            (thumb_pos[0] + text_size[0] + 5, thumb_pos[1] - 10),
                            (0, 0, 0), -1)
                cv2.putText(frame, text, (thumb_pos[0], thumb_pos[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def audio_playback_loop(self):
        """Optimized continuous audio playback loop"""
        print("Starting optimized audio playback loop...")
        channel_index = 0
        max_channels = pygame.mixer.get_num_channels()
        
        while self.audio_playing:
            try:
                # Get next processed chunk
                chunk = self.audio_processor.get_next_chunk()
                
                if chunk is not None and len(chunk) > 0:
                    # Convert to pygame format with better precision
                    chunk = np.clip(chunk, -1.0, 1.0)
                    chunk = (chunk * 32767).astype(np.int16)
                    
                    # Use round-robin channel assignment for smoother playback
                    try:
                        sound = pygame.sndarray.make_sound(chunk)
                        pygame.mixer.Channel(channel_index).play(sound)
                        channel_index = (channel_index + 1) % max_channels
                    except Exception as e:
                        print(f"Pygame sound error: {e}")
                        time.sleep(0.005)
                
                # Faster update rate for smoother audio
                time.sleep(0.01)  # ~100 FPS audio updates
                
            except Exception as e:
                print(f"Audio playback error: {e}")
                time.sleep(0.05)
        
        print("Audio playback loop stopped")
    
    def load_audio(self, audio_file):
        """Load audio file with better error handling"""
        try:
            # Load audio with consistent sample rate
            self.original_audio, self.original_sr = sf.read(audio_file)
            
            # Convert to mono if stereo
            if len(self.original_audio.shape) > 1:
                self.original_audio = np.mean(self.original_audio, axis=1)
            
            # Resample to 22050 Hz for consistency
            if self.original_sr != 22050:
                from scipy.signal import resample
                target_length = int(len(self.original_audio) * 22050 / self.original_sr)
                self.original_audio = resample(self.original_audio, target_length)
                self.original_sr = 22050
            
            # Normalize audio
            max_val = np.max(np.abs(self.original_audio))
            if max_val > 0:
                self.original_audio = self.original_audio / max_val * 0.8
            
            self.audio_processor.load_audio(self.original_audio, self.original_sr)
            print(f"Audio loaded: {len(self.original_audio)} samples at {self.original_sr} Hz")
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
        self.audio_thread = threading.Thread(target=self.audio_playback_loop, daemon=True)
        self.audio_thread.start()
        return True
    
    def stop_audio(self):
        """Stop audio playback"""
        self.audio_playing = False
        pygame.mixer.stop()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
    
    def update_visualization(self):
        """Update real-time visualization with FPS display"""
        current_time = time.time()
        
        # Calculate FPS
        self.frame_count += 1
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
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
            
            self.ax1.plot(times, pitches, 'b-', linewidth=2, label='Pitch Factor', alpha=0.8)
            self.ax1.set_ylabel('Pitch Factor')
            self.ax1.set_title(f'Real-time Audio Modifications (FPS: {self.fps:.1f})')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            self.ax1.set_ylim(self.pitch_range[0] - 0.1, self.pitch_range[1] + 0.1)
            
            self.ax2.plot(times, tempos, 'r-', linewidth=2, label='Tempo Factor', alpha=0.8)
            self.ax2.set_ylabel('Tempo Factor')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            self.ax2.set_ylim(self.tempo_range[0] - 0.1, self.tempo_range[1] + 0.1)
            
            left_distances = [self.left_hand_distance] * len(times)
            right_distances = [self.right_hand_distance] * len(times)
            
            self.ax3.plot(times, left_distances, 'g-', linewidth=2, label='Left Hand Distance', alpha=0.7)
            self.ax3.plot(times, right_distances, 'm-', linewidth=2, label='Right Hand Distance', alpha=0.7)
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
        """Start camera capture with better settings"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            
            # Optimize camera settings for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            self.camera_running = True
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        """Get processed camera frame with performance optimizations"""
        if not self.cap or not self.camera_running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        processed_frame = self.process_hands(frame)
        
        # Enhanced info display
        info_text = [
            f"Pitch Factor: {self.pitch_factor:.2f} (Left Hand)",
            f"Tempo Factor: {self.tempo_factor:.2f} (Right Hand)", 
            f"Left Distance: {self.left_hand_distance:.3f}",
            f"Right Distance: {self.right_hand_distance:.3f}",
            f"Audio Playing: {'Yes' if self.audio_playing else 'No'}",
            f"FPS: {self.fps:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            # Background for better visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(processed_frame, 
                        (5, 25 + i * 30), 
                        (15 + text_size[0], 45 + i * 30),
                        (0, 0, 0), -1)
            cv2.putText(processed_frame, text, (10, 40 + i * 30), 
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
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
            'camera_running': controller.camera_running,
            'fps': controller.fps
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
    print("Starting Optimized Real-time Hand Gesture Audio Controller Backend...")
    print("Using pyrubberband for high-quality real-time audio processing")
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