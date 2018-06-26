import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
#import cv2
import os
import sys
#import datetime
#import traceback
#import numpy as np
#from CropImagesFunctions import cif_step1, cif_step2, cif_step3, cif_isoutlier

app = Flask(__name__,static_folder=None)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')   
    
@app.route('/upload', methods=['POST'])
def upload_file():
    #read image file string data
    file = request.files['image']
    #filestr = request.files['image'].read()
    #convert string data to numpy array
    #npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    #img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    #step1 = cif_step1(img)
    #step2 = cif_step2(step1)
    #step3 = cif_step3(step2)
    
    filename='foto.jpg'
    f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #print f, UPLOAD_FOLDER
    #cv2.imwrite(f, step3)
    file.save(f)
    return redirect(url_for('uploaded_file', filename=filename))
    
@app.route('/show/<filename>')
def uploaded_file(filename):
    return render_template('index.html', filename=filename, init=True)
    
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, cache_timeout = 1)
    

@app.route('/retry', methods=["GET", "POST"])
def retry_():
    
    return render_template('index.html', init=False)

    
