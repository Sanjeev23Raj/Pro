from flask import Flask, request, jsonify
import pickle
import numpy as np

# 1️⃣ Initialize Flask App
app = Flask(__name__)

# 2️⃣ Load the Model
MODEL_PATH = r"C:\\Users\\sanjeev\\OneDrive\\Desktop\\pro\\model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# 3️⃣ Root Route for Confirmation
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running!"})

# 4️⃣ Define API Endpoint (POST Method)
@app.route('/predict', methods=['POST'])
def predict_post():
    try:
        data = request.json.get('features')
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid input. Expected {'features': [values]}"}), 400

        input_data = np.array(data).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 5️⃣ Define API Endpoint (GET Method)
@app.route('/predict', methods=['GET'])
def predict_get():
    try:
        data = request.args.getlist('features')

        if not data:
            return jsonify({"error": "No features provided. Use ?features=value&features=value"}), 400

        input_data = np.array([float(i) for i in data]).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 6️⃣ Run Flask Server
if __name__ == '__main__':
    app.run(debug=True)