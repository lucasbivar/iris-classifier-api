from flask import Flask
from flask import request, abort, jsonify
import numpy as np
from tensorflow.keras.models import model_from_json
import os
from flask_cors import CORS


app = Flask(__name__)

CORS(app, resources={
    r"/":{
        "origins": "*",
    }
})

@app.route('/api/predict', methods=['POST'])
def predictClass():
    if not request.json:
        abort(400)
    if "sepal_length" not in request.json:
        abort(400)
    if "sepal_width" not in request.json:
        abort(400)
    if "petal_length" not in request.json:
        abort(400)
    if "petal_width" not in request.json:
        abort(400)

    sepal_length = request.json["sepal_length"]
    sepal_width = request.json["sepal_width"]
    petal_length = request.json["petal_length"]
    petal_width = request.json["petal_width"]

    try:
        model_file = open("model_iris.json", "r")
        network_structure = model_file.read()
        model_file.close()

        model = model_from_json(network_structure)
        model.load_weights("model_weights.h5")
    except:
        abort(500) 

    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    result = model.predict(user_data)[0]
    iris_class_number = np.argmax(result)

    if(iris_class_number == 0):
        iris_class = "setosa"
    elif (iris_class_number == 1):
        iris_class = "versicolor"
    else:
        iris_class = "virginica"
      
    return jsonify({'type': iris_class}), 200


@app.route("/", methods=["GET"])
def home():
    return "IRIS CLASSIFIER - API", 200


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  response.headers.add('Access-Control-Allow-Credentials', 'true')
  return response

#teste localhost
# if __name__ == '__main__':
#     app.run(debug=True)


#teste heroku
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)