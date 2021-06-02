from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
import requests
import atexit
import os
import json
import cv2
import numpy as np
from watson_machine_learning_client import WatsonMachineLearningAPIClient
import math
import PIL
from PIL import Image

app = Flask(__name__, static_url_path='')
video = cv2.VideoCapture(0)

API_KEY = "dDmZjYiI1A4i_GS-hLYE3LYOukjuzY5DDmN7UNmAxq5L"
location = 'us-south'

model_deployment_endpoint_url    = "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/71b9c30d-be3e-43d0-8d2e-c1d7e3dc7443/predictions"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', 
data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif "CLOUDANT_URL" in os.environ:
    client = Cloudant(os.environ['CLOUDANT_USERNAME'], os.environ['CLOUDANT_PASSWORD'], url=os.environ['CLOUDANT_URL'], connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

@app.route('/')
def root():
    return app.send_static_file('index.html')

# /* Endpoint to greet and add a new visitor to database.
# * Send a POST request to localhost:8000/api/visitors with body
# * {
# *     "name": "Bob"
# * }
# */
@app.route('/api/visitors', methods=['GET'])
def get_visitor():
    if client:
        return jsonify(list(map(lambda doc: doc['name'], db)))
    else:
        print('No database')
        return jsonify([])

# /**
#  * Endpoint to get a JSON array of all the visitors in the database
#  * REST API example:
#  * <code>
#  * GET http://localhost:8000/api/visitors
#  * </code>
#  *
#  * Response:
#  * [ "Bob", "Jane" ]
#  * @return An array of all the visitor names
#  */
@app.route('/api/visitors', methods=['POST'])
def put_visitor():
    user = request.json['name']
    data = {'name':user}
    if client:
        my_document = db.create_document(data)
        data['_id'] = my_document['_id']
        return jsonify(data)
    else:
        print('No database')
        return jsonify(data)

@atexit.register
def shutdown():
    if client:
        client.disconnect()

@app.route('/api/takeimage', methods = ['POST'])
def takeimage():
    name = request.form['name']
    _, frame = video.read()
    cv2.imwrite(f'{name}.jpg', frame)
    img = tf.keras.preprocessing.image.load_img(f'{name}.jpg', target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    payload_scoring =  {'input_data': [{'values': x.tolist()}]}
    url = 'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/3f6f7487-83bb-4b30-b9e0-e761961b6f75/predictions?version=2021-06-02'
    response_scoring = requests.post(url, json=payload_scoring, 
        headers={'Authorization': 'Bearer ' + mltoken})
    return response_scoring.json()

def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
