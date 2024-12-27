from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("activity_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

sensor_data = [
            float(request.form["sensor1"]),
            float(request.form["sensor2"]),
            float(request.form["sensor3"]),
            float(request.form["sensor4"]),
            float(request.form["sensor5"]),
            float(request.form["sensor6"]),
            float(request.form["sensor7"]),
            float(request.form["sensor8"]),
            float(request.form["sensor9"])
        ]
        
        # Predict activity
        prediction = model.predict([sensor_data])[0]
        activity = prediction  # Replace with actual activity labels if needed

        return render_template("index.html", prediction_text=f"Predicted Activity: {activity}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
