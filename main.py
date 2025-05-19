import cv2
import mediapipe as mp
import math
import librosa
import sounddevice as sd
import pyrubberband as pyrb
import numpy as np
from queue import Queue, Empty

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

prev_pitch = 0
prev_rate = 1.0
alpha = 0.31  # Smoothing factor: 0 < alpha < 1

pitch_queue = Queue(maxsize=1)
rate_queue = Queue(maxsize=1)

# Load a sample audio clip
y_full, sr = librosa.load(librosa.example('brahms'), mono=True)
chunk_size = 1024
# chunk_size = 2048
play_ptr = 0


last_pitch = 0
last_rate = 1.0

def audio_callback(outdata, frames, time, status):
    global play_ptr
    global last_pitch, last_rate

    try:
        pitch = pitch_queue.get_nowait()
        last_pitch = 0.2 * pitch + 0.8 * last_pitch  # Smooth
    except Empty:
        pitch = last_pitch

    try:
        rate = rate_queue.get_nowait()
        last_rate = 0.2 * rate + 0.8 * last_rate  # Smooth
    except Empty:
        rate = last_rate

    chunk = y_full[play_ptr:play_ptr+chunk_size]
    if len(chunk) < chunk_size:
        play_ptr = 0
    play_ptr += chunk_size

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

stream = sd.OutputStream(
    samplerate=sr,
    blocksize=chunk_size,
    channels=1,
    callback=audio_callback
)
stream.start()

with mp_hands.Hands(
    model_complexity=0,
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame.flags.writeable = True 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        pitch = 0
        rate = 1.0

        if results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks, start=1):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(idx_tip.x * w), int(idx_tip.y * h)

                dist = math.hypot(x2 - x1, y2 - y1) / 100

                if hand_no == 1:
                    raw_pitch = float(np.interp(dist, [1, 5], [-12, 12]))
                    pitch = alpha * raw_pitch + (1 - alpha) * prev_pitch
                    prev_pitch = pitch

                    pitch_queue.queue.clear()
                    pitch_queue.put(pitch)
                else:
                    raw_rate = float(np.interp(dist, [1, 5], [0.5, 2.0]))
                    rate = alpha * raw_rate + (1 - alpha) * prev_rate
                    prev_rate = rate

                    rate_queue.queue.clear()
                    rate_queue.put(rate)

                cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
                cv2.putText(frame,
                    f"Hand {hand_no} Dist: {dist:.1f}px",
                    (10, 30 * hand_no),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow('Hand-Controlled Pitch & Tempo', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()