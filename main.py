import soundfile
from multiprocessing import Process, Queue, Value

def changePitch(steps):
    new_y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    soundfile.write("pitchShifted.wav", new_y, sr,)

def changeSpeed():
    new_y = librosa.effects.time_stretch(y, rate=1.5)
    soundfile.write("speedChanged.wav", new_y, sr,)


def generator(sound_buffer: Queue, factor: Value, offset: int=500):
    changePitch(12)
    changeSpeed()

filename = librosa.example("brahms")
y, sr = librosa.load(filename)
sound_buffer = Queue(maxsize=2)
sound_factor = Value('d', 1)
sound_offset = 500

def main():
    p = Process(generator, args=(sound_buffer, sound_factor, sound_offset), daemon=True)