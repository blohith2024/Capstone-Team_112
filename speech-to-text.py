import numpy as np
import sounddevice as sd
import wave
import os
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # Sample rate in Hz
CHANNELS = 1  # Mono audio
DTYPE = 'int16'  # 16-bit audio format

# Function to record audio
def record_audio(file_path, duration):
    print(f"Recording for {duration} seconds...")
    frames = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    
    # Save to WAV file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(frames.tobytes())

# Function to transcribe audio
def transcribe_audio(model, file_path):
    segments, _ = model.transcribe(file_path)
    transcription = " ".join(segment.text for segment in segments)
    return transcription


def main():
    model_size = "large-v1"  
    model = WhisperModel(model_size, device="cpu", compute_type="float32")
    
    duration = int(input("Enter recording duration (seconds): "))
    audio_file = "recorded_audio.wav"
    
    record_audio(audio_file, duration)
    transcription = transcribe_audio(model, audio_file)
    
    print("\nTranscription:")
    print(transcription)
    
    with open("transcription_log.txt", "w") as log_file:
        log_file.write(transcription)

if __name__ == "__main__":
    main()
