'''
Simple inference server.

Summary
-------
A very primitive inference server is realized.
It exposes our created models as a web API.
One can send a request to the running server via the command
"curl -X POST -F image=@test.jpg http://localhost:5000/predict".
Of course, one can query also programmatically.

'''

import io

from PIL import Image
from flask import Flask, jsonify, request

from models import OLXModel


model = OLXModel('weights.pt')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image'].read()
            image = Image.open(io.BytesIO(image))
            # image = Image.open(request.files['image'])

            blurriness, clockness, is_clock = model(
                image,
                transform_mode='full',
                threshold=0.5
            )

            prediction = {
                'blurriness': blurriness,
                'clockness': clockness,
                'isclock': is_clock
            }

            return jsonify(prediction)


if __name__ == '__main__':

    app.run()

