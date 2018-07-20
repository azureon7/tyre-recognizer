import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, Markup
import cv2
import sys
import traceback
import numpy as np
from CropImagesFunctions import cif_logpolar_manual, cif_logpolar_manual_90, cif_logpolar_auto, cif_logpolar_auto_90, cif_step1, cif_crop, image_resize, cif_preproc
from ScrapingFunctions import Parser
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

app.config['OCR']=False

#================================================================
# HOMEPAGE , ABOUT , RESULTS , TRY
#================================================================

@app.route('/')
def homepage():
    return render_template('index.html')   
    
@app.route('/method')
def method():
    return render_template('method.html')   
    
@app.route('/author')
def author():
    return render_template('author.html')   
    
@app.route('/try')
def try__():
    app.config['OCR']=False
    return render_template('try.html', enabledl=True, preview = True, OCR=False)
    
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
    app.config['OCR']=False
    return render_template('try.html', init=False, enabledl=True, preview = True, OCR=False)
    
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
    app.config['OCR']='bridgestone'
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
    app.config['OCR']='continental'
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
    app.config['OCR']='michelin'
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
    app.config['OCR']='pirelli'
    return redirect(url_for('uploaded_file', filename=filename_preview))    
    
    
#================================================================
# IMAGE CROPPING
#================================================================

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
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
        img_small = image_resize(img.copy(), width=const0)
        filename='step0.jpg'
        filename_preview='step0_preview.jpg'
        f1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f2 = os.path.join(app.config['UPLOAD_FOLDER'], filename_preview)
        cv2.imwrite(f1, img)
        cv2.imwrite(f2, img_small)
        return redirect(url_for('uploaded_file', filename=filename_preview))
    except:
        app.config['OCR']=False
        return render_template('try.html', enabledl=True, preview = True, OCR=False)

    
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
    
    ocr_out = ''
    param1 = 'e.g. 205'
    param2 = 'e.g. 55'
    param3 = 'e.g. 16'
    if app.config['OCR'] != False:
        ocr_results = {'bridgestone': '<p>REDWEAL\nWITHOUT CONTACTO\n\nIKAK LOAD CH 012355\nAT 850Tat1 KAX PECAS\nTUBI LESS RADIAL\nSPAR\n7516Y <span class="measur">185 65 R15 </span>SSE\nPYSWZ\nE\nOUTSIDE\nPLIECETREAD POLYESTER - 2 STEL - 1 POLYFOTTE\nSIDE ALLIPOLYESTER\nBCLX U9F (0813)\n9) 0290 74 52WRE\nE\n00933413\nD EG GODT NG PET PPSELT HEAD\nDUPIETEET\nSED DUE PEOPGE\nOLO\nNEOS DOU D\nOLD TOUTES\n-\n</p>',\
                       'continental': '<p>Continental\n-e as .\nConfinoles\nTREAOTEAR 280\nTRACTION\nA\nTEMPERATURE\ntinental\n <span class="measur">205/55R16 </span>V.\nContiPremiumContact 5\nNOM\nDOT GYOF D7L5 1816\nCONTINENTAL\nL-99812S-2328112328473\ncontinental-tires.com\n82\nMAX INFLATION PRESSURE 350 KPA (1 PSD\nMAX LOAD 615 KG (1356 LB)\nPY\nSTELA\n</p>',\
                       'michelin': '<p>maiores\n175/65 R\nWARNING\n <span class="measur">175/65R14 </span>\n327\nR\nA\nA\nSDPLES\nIG HELINO TUBELES RADIAL X\nBIREWOLLPLY\nCena\nMICHELINO TUBELES3 RADIAL\nFOLESTE\n122502 52\nbesparende\n</p>',\
                       'pirelli': '<p>Per la\n7.com\nwww.pirelli\nESTERNO AUSSEN\nEXTERIEUR OUTER\nASA CANADA WA LOLEN ONLI\nLO 615 1356 051\n409001\nP <span class="measur">205/55 R16 </span>\nSTANDARD LOAD\n1316)\nRADIAL\nTUBELESS\n21.129SZ WRI\n6253353\nLIEGEZWE COQ\nOLETS\n1089978\nAG9978\n2012\n</p>'}
        diz_param = {'bridgestone': ['185', '65', '15'],\
                     'continental': ['205', '55', '16'],\
                     'michelin': ['175', '65', '14'],\
                     'pirelli': ['205', '55', '16']}
        param1 = diz_param[app.config['OCR']][0]
        param2 = diz_param[app.config['OCR']][1]
        param3 = diz_param[app.config['OCR']][2]
        ocr_out = ocr_results[app.config['OCR']]
    
    return render_template('try.html', filename='step2.jpg', init=False, crop1 = False, crop2=True,\
                            pred = predizione, prob = prob, downloadenabled=app.config['DOWNLOAD'], OCR=app.config['OCR'], ocr_text = ocr_out,
                            marca = predizione, param1 = param1, param2 = param2, param3 = param3)
    

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
    
#================================================================
# WEB SCRAPING
#================================================================    

@app.route('/scraping', methods=["GET", "POST"])
def scraping_():
    
    try:
        Marca = request.form['marchio']
        p1 = request.form['param1'].strip()
        p2 = request.form['param2'].strip()
        p3 = request.form['param3'].strip()
        
        scraping_results = Parser(Marca, p1, p2, p3)
        
        return render_template('try.html', init=False, crop1 = False, crop2=False, scrap_area=True, OCR=False,\
                               nome1=scraping_results[0]['nome'], url1=scraping_results[0]['url'],\
                               img1=scraping_results[0]['img'], prezzo1=scraping_results[0]['price'],\
                               nome2=scraping_results[1]['nome'], url2=scraping_results[1]['url'],\
                               img2=scraping_results[1]['img'], prezzo2=scraping_results[1]['price'])
                               
    except:
        return render_template('try.html', error2=True)




