import cv2
import mediapipe as mp
import math
import librosa
import sounddevice as sd
import pyrubberband as pyrb
import numpy as np
import time
from queue import Queue, Empty

# Setup MediaPipe and queues
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
pitch_queue = Queue(maxsize=1)
rate_queue = Queue(maxsize=1)

# Load sample audio
y_full, sr = librosa.load(librosa.example('brahms'), mono=True)
chunk_size = 1024
play_ptr = 0

# Audio callback
def audio_callback(outdata, frames, time_info, status):
    global play_ptr
    chunk = y_full[play_ptr:play_ptr+chunk_size]

    if len(chunk) < chunk_size:
        play_ptr = 0
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

# Start audio stream
stream = sd.OutputStream(
    samplerate=sr,
    blocksize=chunk_size,
    channels=1,
    callback=audio_callback
)
stream.start()

# Utility: Draw bar
def draw_bar(image, x, y, value, min_val, max_val, color, label):
    bar_length = 200
    norm_val = np.clip((value - min_val) / (max_val - min_val), 0, 1)
    filled_length = int(norm_val * bar_length)

    cv2.rectangle(image, (x, y), (x + bar_length, y + 20), (40, 40, 40), -1)
    cv2.rectangle(image, (x, y), (x + filled_length, y + 20), color, -1)
    cv2.putText(image, f"{label}: {value:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# FPS setup
prev_time = time.time()

# Hand tracking
with mp_hands.Hands(
    model_complexity=0,
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    cv2.namedWindow("Hand-Controlled Pitch & Tempo", cv2.WINDOW_GUI_EXPANDED)

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

        overlay = frame.copy()
        status_msg = "🤚 No hands detected"
        colors = [(0, 255, 0), (255, 0, 0)]  # Green, Blue

        if results.multi_hand_landmarks:
            status_msg = "🎵 Adjusting Pitch & Tempo"

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
                    pitch = float(np.interp(dist, [1, 5], [-12, 12]))
                    pitch_queue.queue.clear()
                    pitch_queue.put(pitch)
                else:
                    rate = float(np.interp(dist, [1, 5], [0.5, 2.0]))
                    rate_queue.queue.clear()
                    rate_queue.put(rate)

                # Draw finger points
                cv2.circle(overlay, (x1, y1), 10, colors[hand_no - 1], -1)
                cv2.circle(overlay, (x2, y2), 10, colors[hand_no - 1], -1)

        # Transparent UI background
        cv2.rectangle(overlay, (0, 0), (350, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Overlay text & bars
        cv2.putText(frame, "🎧 Real-Time Audio Shifter", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status_msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        draw_bar(frame, 10, 90, pitch, -12, 12, (0, 255, 0), "Pitch")
        draw_bar(frame, 10, 130, rate, 0.5, 2.0, (255, 0, 0), "Tempo")

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Hand-Controlled Pitch & Tempo', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()
