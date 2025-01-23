import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyaudio
import wave

# 1. Record Audio from Microphone
def record_audio(filename="baby_cry.wav"):
    p = pyaudio.PyAudio()
    rate = 16000
    chunk = 1024
    channels = 1
    format = pyaudio.paInt16
    seconds = 5

    stream = p.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

# 2. Extract MFCC Features
def extract_features(filename):
    y, sr = librosa.load(filename, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# 3. Train a Model (For Demonstration, using RandomForest)
def train_model():
    # Assuming you have pre-labeled dataset (features and labels)
    # features = np.array([...])  # Extracted MFCC features from training data
    # labels = np.array([...])    # Labels for each audio sample
    
    # Example dummy data (replace with real data)
    features = np.random.rand(100, 13)  # 100 samples with 13 MFCC features each
    labels = np.random.choice(['hungry', 'tired', 'discomfort'], size=100)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Initialize and train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    return model

# 4. Predict Crying Reason (Based on the New Recorded Audio)
def predict_crying_reason(model, filename="baby_cry.wav"):
    features = extract_features(filename).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Start the Flask app
if __name__ == "__main__":
       # Record the crying sound from the sensor
    record_audio()

    # Train the model (you only need to do this once, and you can save the model)
    model = train_model()

    # Predict the reason for the crying based on the recorded sound
    reason = predict_crying_reason(model)
    print(f"The detected reason for the crying is: {reason}")