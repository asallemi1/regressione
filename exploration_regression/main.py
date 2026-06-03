from flask import Flask, jsonify, request
from it.akron.data_explorer import PuliziaDataset
from flask import send_file
from it.akron.regressione import Regressione
import pandas as pd

app = Flask(__name__)


@app.route('/new_veicle', methods=['POST'])
def new_prediction():

    data = request.get_json()

    df_new = pd.DataFrame([{
        "displacement": data.get("displacement"),
        "cylinders": data.get("cylinders"),
        "horsepower": data.get("horsepower"),
        "weight": data.get("weight"),
        "acceleration": data.get("acceleration"),
        "model_year": data.get("model_year"),
        "origin": data.get("origin")
    }])

    df = PuliziaDataset()
    X, y = df.trova_outliers()

    model = Regressione(X, y, metodo=request.args.get("metodo"))
    model.train()

    prediction = model.new_pred(df_new)

    return jsonify({"prediction": float(prediction[0])})

@app.route('/mse')
def mse():
    metodo = request.args.get("metodo")
    df = PuliziaDataset()
    X, y = df.trova_outliers()
    model = Regressione(X, y, metodo=request.args.get("metodo"))
    mse = model.train()
    return jsonify({"mse": mse})


@app.route('/na')
def individua_na():
    df = PuliziaDataset()
    null, perc = df.individua_na()
    return jsonify({
        "null values": null.to_dict(),
        "percentage": perc.to_dict()
    })


@app.route('/pca')
def pca():
    df = PuliziaDataset()
    img = df.applica_pca()
    return send_file(img, mimetype='image/png')

@app.route('/hist')
def hist():
    df = PuliziaDataset()
    img = df.grafici_distribuzioni()
    return send_file(img, mimetype='image/png')

@app.route('/boxplot')
def boxplot():
    df = PuliziaDataset()
    img = df.grafici_boxplot()
    return send_file(img, mimetype='image/png')

@app.route('/corr')
def correlation():
    df = PuliziaDataset()
    corr = df.matrice_correlazione()

    return jsonify(corr.to_dict())

@app.route('/testJB')
def testJB():
    df = PuliziaDataset()
    test = df.jarque_bera_test()
    return jsonify(test)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
