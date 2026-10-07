from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained NLP Pipeline
with open("resume_reader_pipeline.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        resume = request.form["resume"]

        # Pipeline automatically performs:
        # Resume → TF-IDF → LinearSVC → Prediction
        prediction = model.predict([resume])[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)