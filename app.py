import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from predict import predict
import numpy as np

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)


dog_breeds = np.load('classes.npy')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return 'Hello'

def predictImg(path):
    result = predict.predict(path)[0]
    temp = result[:]
    temp = temp.argsort()[-3:][::-1]
    return {
        'top1': {
            'name': dog_breeds[temp[0]],
            'percent': "{:.2f}".format(result[temp[0]] * 100)
        },
        'top2' : {
            'name': dog_breeds[temp[1]], 
            'percent': "{:.2f}".format(result[temp[1]] * 100)
        },
        'top3' : {
            'name': dog_breeds[temp[2]], 
            'percent': "{:.2f}".format(result[temp[2]] * 100)
        },
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return {
                'data': None,
                'message' : 'No file sent'
            }
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return {
                'data': None,
                'message' : 'No file sent'
            }
        if not allowed_file(file.filename):
            return {
                'data': None,
                'message': 'Wrong format'
            }
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            result = predictImg(img_path)
            return {
                'data': result,
                'message': 'Done'
            }   


if __name__ == "__main__":
    app.run()