from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("smart_lender_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    result_class = ""

    if request.method == "POST":

        features = [
            float(request.form["Gender"]),
            float(request.form["Married"]),
            float(request.form["Dependents"]),
            float(request.form["Education"]),
            float(request.form["Self_Employed"]),
            float(request.form["ApplicantIncome"]),
            float(request.form["CoapplicantIncome"]),
            float(request.form["LoanAmount"]),
            float(request.form["Loan_Amount_Term"]),
            float(request.form["Credit_History"]),
            float(request.form["Property_Area"])
        ]

        prediction_result = model.predict([features])

        if prediction_result[0] == 1:
            prediction = "Loan Approved"
            result_class = "approved"
        else:
            prediction = "Loan Rejected"
            result_class = "rejected"

    return render_template(
        "index.html",
        prediction=prediction,
        result_class=result_class
    )
if __name__ == "__main__":
    app.run(debug=True)