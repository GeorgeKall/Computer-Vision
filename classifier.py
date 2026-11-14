import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("data/hand_gestures.csv")

# Separate features (landmarks) and labels
X = data.drop("label", axis=1)
y = data["label"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Test it
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model trained! Accuracy: {acc * 100:.2f}%")

# Save the model
with open("data/hand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to data/hand_model.pkl")
