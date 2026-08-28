import numpy as np
from flask import Flask, request, render_template
import pickle
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
model = pickle.load(open(os.path.join(base_dir, "ufo-model.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_lectures = [np.array(int_features)]
    prediction = model.predict(final_lectures)

    output = prediction[0]

    countries = ["Australia", "Canada" , "Germany", "UK", "US"]
    return render_template(
        "index.html" , prediction_text = "Likely country: {}".format(countries[output])
    )
if __name__ == "__main__":
    app.run(debug = True)