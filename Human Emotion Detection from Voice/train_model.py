# train_model.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from feature_extraction import extract_features

ravdess_data= r"D:\Download\Audio_Speech_Actors_01-24"

emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
# ✅ Confirm directory exists and is not empty
if not os.path.isdir(ravdess_data):
    print("❌ The path is not a valid directory:", ravdess_data)
else:
    print("✅ Directory exists. Number of files:", len(os.listdir(ravdess_data)))

# ✅ Walk through all subfolders to collect .wav files
X, y = [], []

# for file in os.listdir(ravdess_data):
#     if file.endswith(".wav"):
#         emotion_code = file.split("-")[2]
#         emotion = emotions.get(emotion_code)
#         if emotion:
#             features = extract_features(os.path.join(ravdess_data, file))
#             X.append(features)
#             y.append(emotion)

# ✅ Walk through all subfolders to collect .wav files
for root, dirs, files in os.walk(ravdess_data):
    for file in files:
        if file.endswith(".wav"):
            try:
                emotion_code = file.split("-")[2]
                emotion = emotions.get(emotion_code)
                if emotion:
                    file_path = os.path.join(root, file)
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
                    else:
                        print(f"⚠️ Skipped {file} — no features extracted.")
                else:
                    print(f"⚠️ Skipped {file} — unknown emotion code.")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# ✅ Check if any features were collected
print(f"✅ Total samples collected: {len(X)}")
if len(X) == 0:
    print("❌ No audio features extracted. Please check your extract_features() function.")
    exit()
    
# ✅ Train/test split  
X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, "emotion_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
