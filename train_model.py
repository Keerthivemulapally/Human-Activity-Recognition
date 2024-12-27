import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset (make sure your dataset has sensor data like accelerometer readings)
data = pd.read_csv("sensor_data.csv")  # Replace with your actual dataset path

# Preprocessing - Feature Engineering (assumes dataset already has 'activity' column for labels)
X = data.drop('activity', axis=1)  # Features (sensor data)
y = data['activity']  # Target (activity labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("activity_model.pkl", "wb") as file:
    pickle.dump(model, file)
