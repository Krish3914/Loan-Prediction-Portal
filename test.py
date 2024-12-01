from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("loanmodel.sav", "rb"))

# Preprocessing function
def preprocess_input(data):
    """Preprocess user input data."""
    ord_enc = OrdinalEncoder()
    data[["Gender", "Married", "Education", "Self_Employed", "Property_Area"]] = ord_enc.fit_transform(
        data[["Gender", "Married", "Education", "Self_Employed", "Property_Area"]]
    )
    return data

# Route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect user inputs
        gender = request.form["gender"]
        married = request.form["married"]
        education = request.form["education"]
        self_employed = request.form["self_employed"]
        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_amount_term = float(request.form["loan_amount_term"])
        credit_history = float(request.form["credit_history"])
        property_area = request.form["property_area"]

        # Create DataFrame from inputs
        user_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        # Preprocess the input data
        processed_data = preprocess_input(user_data)

        # Predict using the model
        prediction = model.predict(processed_data)
        result = "Approved" if prediction[0] == 1 else "Not Approved"

        return render_template("result.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
