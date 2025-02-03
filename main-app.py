from flask import Flask, render_template, request  

import pickle
# Flask App
app = Flask(__name__)
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get("name")
    age = int(request.form["age"])
    gender = label_encoder.transform([request.form["gender"]])[0]
    income = float(request.form["income"])
    
    input_data = scaler.transform([[age, gender, income]])
    # prediction = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)
    
    result = f"Hi {name}, your loan is {'approved' if prediction == 1 else 'rejected'}"
    return render_template('index.html', output=result)

if __name__ == "__main__":
    app.run(debug=True)

print("Model and Flask App ready!")