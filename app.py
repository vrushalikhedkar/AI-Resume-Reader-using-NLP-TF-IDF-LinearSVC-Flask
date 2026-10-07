from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


# Load trained model
with open("resume_reader_model.pkl", "rb") as file:
    model = pickle.load(file)


# Load TF-IDF vectorizer
with open("resume_reader_tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        resume = request.form["resume"]

        resume_tfidf = tfidf.transform([resume])

        prediction = model.predict(resume_tfidf)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)