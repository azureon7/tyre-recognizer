import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import cv2
import sys
import traceback
import numpy as np
from CropImagesFunctions import cif_logpolar_manual, cif_logpolar_manual_90, cif_logpolar_auto, cif_logpolar_auto_90, cif_step1, cif_crop, image_resize, cif_preproc
import label_image
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

const0 = 600
input_layer = "Mul"
output_layer = "final_result"
input_name = "import/" + input_layer
output_name = "import/" + output_layer
model = label_image.load_graph('retrained_graph.pb')
labels = ['Bridgestone', 'Continental', 'Michelin', 'Pirelli']

app.config['DOWNLOAD']=False

#================================================================
# HOMEPAGE , ABOUT , RESULTS , TRY
#================================================================

@app.route('/')
def homepage():
    return render_template('index.html')   
    
@app.route('/about')
def about():
    return render_template('about.html')   
    
@app.route('/results')
def results():
    return render_template('results.html')   
    
@app.route('/try')
def try__():
    return render_template('try.html', enabledl=True, preview = True)   
    
#================================================================
# METHODS
#================================================================

@app.route('/show/<filename>')
def uploaded_file(filename):
    return render_template('try.html', filename=filename, init=True, preview = False)
   
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, cache_timeout = 1)
     
@app.route('/retry', methods=["GET", "POST"])
def retry_():
    return render_template('try.html', init=False, enabledl=True, preview = True)
    
#================================================================
# STARTING FROM IMAGE THUMBNAILS
#================================================================
    
@app.route('/bridgestone')
def bridgestone():
    img = cv2.imread('static/imgs/bridgestone_example.jpg')
    img_small = image_resize(img.copy(), width=const0)
    filename='step0.jpg'
    filename_preview='step0_preview.jpg'
    f1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f2 = os.path.join(app.config['UPLOAD_FOLDER'], filename_preview)
    cv2.imwrite(f1, img)
    cv2.imwrite(f2, img_small)
    return redirect(url_for('uploaded_file', filename=filename_preview))


@app.route('/continental')
def continental():
    img = cv2.imread('static/imgs/continental_example.jpg')
    img_small = image_resize(img.copy(), width=const0)
    filename='step0.jpg'
    filename_preview='step0_preview.jpg'
    f1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f2 = os.path.join(app.config['UPLOAD_FOLDER'], filename_preview)
    cv2.imwrite(f1, img)
    cv2.imwrite(f2, img_small)
    return redirect(url_for('uploaded_file', filename=filename_preview))

@app.route('/michelin')
def michelin():
    img = cv2.imread('static/imgs/michelin_example.jpg')
    img_small = image_resize(img.copy(), width=const0)
    filename='step0.jpg'
    filename_preview='step0_preview.jpg'
    f1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f2 = os.path.join(app.config['UPLOAD_FOLDER'], filename_preview)
    cv2.imwrite(f1, img)
    cv2.imwrite(f2, img_small)
    return redirect(url_for('uploaded_file', filename=filename_preview))

@app.route('/pirelli')
def pirelli():
    img = cv2.imread('static/imgs/pirelli_example.jpg')
    img_small = image_resize(img.copy(), width=const0)
    filename='step0.jpg'
    filename_preview='step0_preview.jpg'
    f1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f2 = os.path.join(app.config['UPLOAD_FOLDER'], filename_preview)
    cv2.imwrite(f1, img)
    cv2.imwrite(f2, img_small)
    return redirect(url_for('uploaded_file', filename=filename_preview))    
    
    
#================================================================
# IMAGE CROPPING
#================================================================

@app.route('/upload', methods=['POST'])
def upload_file():
    #read image file string data
    #file = request.files['image']
    filestr = request.files['image'].read()
    app.config['image_name'] = request.files['image'].filename
    value = request.form.getlist('enabledownload')
    if value:
        app.config['DOWNLOAD'] = True
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    try:
        img_small = image_resize(img.copy(), width=const0)
        filename='step0.jpg'
        filename_preview='step0_preview.jpg'
        f1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f2 = os.path.join(app.config['UPLOAD_FOLDER'], filename_preview)
        cv2.imwrite(f1, img)
        cv2.imwrite(f2, img_small)
        return redirect(url_for('uploaded_file', filename=filename_preview))
    except:
        return render_template('try.html', enabledl=False )

    
@app.route('/crop_step1', methods=["GET", "POST"])
def crop_step1():
    img_raw = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'step0.jpg'))
    const2 = img_raw.shape[1]*1.0 / const0
    #------------------
    try: 
        x1 = int(int(request.form['x1'])*const2)
        x2 = int(int(request.form['x2'])*const2)
        y1 = int(int(request.form['y1'])*const2)
        y2 = int(int(request.form['y2'])*const2)
        flag1 = True
    except:
        flag1 = False
    #------------------
    # cropping - step 1 
    # procedura manuale
    if flag1:
        img_raw = img_raw[y1:y2,x1:x2]
        img_raw = cv2.resize(img_raw, (6000, 6000))
        img = cif_logpolar_manual(img_raw)
        img_90 = cif_logpolar_manual_90(img_raw)
    # procedura automatica
    else:
        img_raw = cif_step1(img_raw)
        img = cif_logpolar_auto(img_raw)
        img_90 = cif_logpolar_auto_90(img_raw)
    #------------------    
    # preprocessing
    img = cif_preproc(img)
    img_90 = cif_preproc(img_90)
    img_small = cv2.resize(img, (const0, 300))
    #------------------
    # saving
    f1 = os.path.join(app.config['UPLOAD_FOLDER'], 'step1.jpg')
    f2 = os.path.join(app.config['UPLOAD_FOLDER'], 'step1_preview.jpg')
    f1_90 = os.path.join(app.config['UPLOAD_FOLDER'], 'step1_90.jpg')
    cv2.imwrite(f1, img)
    cv2.imwrite(f2, img_small)
    cv2.imwrite(f1_90, img_90)
    #------------------   
    return redirect(url_for('show_crop1', filename='step1_preview.jpg'))

@app.route('/show_crop1/<filename>')
def show_crop1(filename):
    return render_template('try.html', filename=filename, init=False, crop1=True)
    

@app.route('/crop_step2', methods=["GET", "POST"])
def crop_step2():
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'step1.jpg'))
    img_90 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'step1_90.jpg'))
    const2 = img.shape[0]*1.0 / 300
    #------------------   
    try:
        y1 = int(int(request.form['y1'])*const2)
        y2 = int(int(request.form['y2'])*const2)
        flag2 = True
    except:
        flag2 = False
    #------------------   
    if flag2:
        img = img[y1:y2,:]
        img_90 = img_90[y1:y2,:]
    else:
        img = cif_crop(img)
        img_90 = cif_crop(img_90)
    #------------------
    # saving
    f1 = os.path.join(app.config['UPLOAD_FOLDER'], 'step2.jpg')
    f1_90 = os.path.join(app.config['UPLOAD_FOLDER'], 'step2_90.jpg')
    cv2.imwrite(f1, img)
    cv2.imwrite(f1_90, img_90)
    #------------------
    return redirect(url_for('show_crop2', filename='step2.jpg'))

    
@app.route('/show_crop2/<filename>')
def show_crop2(filename):
    # prima predizione 
    filename=os.path.join(app.config['UPLOAD_FOLDER'], 'step2.jpg')
    tensor = label_image.read_tensor_from_image_file(filename)
    input_operation = model.get_operation_by_name(input_name)
    output_operation = model.get_operation_by_name(output_name)
    with tf.Session(graph=model) as sess:
         results = sess.run(output_operation.outputs[0],
                            {input_operation.outputs[0]: tensor})
    results = list(np.squeeze(results))    
    print(results)
    index_ = results.index(max(results))
    predizione_1 = str(labels[index_])
    prob_1 = results[index_]
    
    # seconda predizione
    filename=os.path.join(app.config['UPLOAD_FOLDER'], 'step2_90.jpg')
    tensor = label_image.read_tensor_from_image_file(filename)
    input_operation = model.get_operation_by_name(input_name)
    output_operation = model.get_operation_by_name(output_name)
    with tf.Session(graph=model) as sess:
         results = sess.run(output_operation.outputs[0],
                            {input_operation.outputs[0]: tensor})
    results = list(np.squeeze(results))    
    print(results)
    index_90 = results.index(max(results))
    predizione_2 = str(labels[index_90])
    prob_2 = results[index_90]
    
    predizione = predizione_1
    prob = round(prob_1,2)
    if prob_2 > prob_1:
        predizione = predizione_2
        prob = round(prob_2,2)
    
    return render_template('try.html', filename='step2.jpg', init=False, crop1 = False, crop2=True,\
                            pred = predizione, prob = prob, downloadenabled=app.config['DOWNLOAD'])
    

@app.route('/download', methods=['POST'])
def return_file():
    nome = request.form['downloadname']
    nome = str(nome[:4].lower()) + '_' + str(app.config['image_name'])
    return send_from_directory(directory='uploads', filename='step2.jpg', as_attachment=True, attachment_filename=nome)
    

@app.route('/download_90', methods=['POST'])
def return_file_90():
    nome = request.form['downloadname_90']
    nome = str(nome[:4].lower()) + '_90_' + str(app.config['image_name'])
    return send_from_directory(directory='uploads', filename='step2_90.jpg', as_attachment=True, attachment_filename=nome)