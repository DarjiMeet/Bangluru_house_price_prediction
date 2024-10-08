from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_name()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    response = jsonify({
        'estimated_price': util.get_price(location,total_sqft,bath,bhk)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



if __name__ == '__main__':
    print('Server is starting...')
    util.load_artifacts()
    app.run()