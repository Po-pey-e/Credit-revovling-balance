from flask import Flask, request, render_template
import numpy as np
import pickle


app = Flask(__name__, template_folder="templates")
model = pickle.load(open('linear.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/output", methods=["POST"])
def output():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    return render_template('output.html', prediction_text=int_features)
