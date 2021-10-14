from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd


def load_pickle(fn):
    with open(fn, 'rb') as f: return pickle.load(f)


app = Flask(__name__)


@app.route("/")
def home(): return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def price_pred():
    model = pickle.load(open("model.pkl", "rb"))
    choice = load_pickle("column_names.pkl")
    # intp = np.array([request.args.get(name) for name in choice])
    intp = np.array([float(x) for x in request.form.values()])
    # intp = np.array([20.0, 90000.0, 1283.0, 1015.0, 472.0])  # testing
    
    pred_price = model.predict(np.expand_dims(intp, axis=0))
    # return jsonify({"House Price: ": str(pred_price)})
    return render_template("index.html", prediction_text=f"House price should be {str(pred_price)}.")



if __name__ == "__main__": app.run(debug=True)