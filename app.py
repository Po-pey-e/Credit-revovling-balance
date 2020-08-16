from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__, template_folder="templates")
model = pickle.load(open('linear.pkl', 'rb'))
# model = pickle.load(open('randomForest.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/output", methods=["POST"])
def output():
    # select_inputs = ("terms", "verification_status", "Purpose", "State",
    #                  "initial_list_status", "application_type")
    # lb_make = LabelEncoder()
    int_features = []
    f = request.form
    for key in f.keys():
        for value in f.getlist(key):
            int_features.append(value)
    #         if(key not in select_inputs):
    #             int_features.append(float(value))
    #         else:
    #             int_features.append(lb_make.fit_transform(value))

    final_features = np.array(int_features, dtype="float64").reshape(1, -1)
    prediction = model.predict(final_features)

    return render_template('output.html', prediction_text=prediction[0][0])
