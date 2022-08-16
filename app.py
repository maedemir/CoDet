"""
Covid-19 Detector UI

@authors: 
   . Nami (@SNamiMod on github)
   . Maedeh (@maedemir on github)

Summer 2022
"""

# 359 covid

from pickle import TRUE
from statistics import mode
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import Mail_Client as m
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

app = Flask(__name__,static_url_path='/assets',
            static_folder='../Flask/assets', 
            template_folder='../Flask')

UPLOAD_FOLDER = './assets/images/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/project.html')
def project():
   return render_template('project.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/detect.html')
def detect():
   return render_template('detect.html')

@app.route('/uploader_ct', methods = ['POST', 'GET'])
def uploader_ct():
   
   if request.method == 'POST':
      if 'file' not in request.files:
         flash('No file part')
         return render_template('detect.html')
      file = request.files['file']
      name = file.filename
      model = request.form.get("model")  
      if file.filename == '':
         flash('No selected file')
         return render_template('detect.html')
      if model == '0':
         flash('No selected model')
         return render_template('detect.html')

      if file:
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
         vgg16 = load_model('models/vgg16/VGG16_CT_model.h5')
         vgg19 = load_model('models/vgg19/VGG19_CT_model.h5')
         inceptionv3 = load_model('models/inceptionv3/InceptionV3_CT_model.h5')

         image = cv2.imread("./assets/images/upload/" + name)
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         image = cv2.resize(image, (224, 224))
         image = np.array(image) / 255.0
         image = np.expand_dims(image, axis=0)

         if model == "1" or model == "4":
            vgg16_predict = vgg16.predict(image)
            probability = vgg16_predict[0]
            if probability[1] > 0.5:
               vgg16_CT_predict = str('%.2f' % (probability[1] * 100) + '% COVID')
            else:
               vgg16_CT_predict = str('%.2f' % ((probability[0]) * 100) + '% NonCOVID')
         
         if model == "2" or model == "4":
            vgg19_predict = vgg19.predict(image)
            probability = vgg19_predict[0]
            if probability[1] > 0.5:
               vgg19_CT_predict = str('%.2f' % (probability[1] * 100) + '% COVID')
            else:
               vgg19_CT_predict = str('%.2f' % ((probability[0]) * 100) + '% NonCOVID')
         
         if model == "3" or model == "4":
            inceptionv3_predict = inceptionv3.predict(image)
            probability = inceptionv3_predict[0]
            if probability[1] > 0.5:
               inceptionv3_CT_predict = str('%.2f' % (probability[1] * 100) + '% COVID')
            else:
               inceptionv3_CT_predict = str('%.2f' % ((probability[0]) * 100) + '% NonCOVID')
         
         if model == "1":
            data=[
               {
                  'model':"Vgg16",
                  'result':vgg16_CT_predict
               }
            ]
         if model == "2":
            data=[
               {
                  'model':"Vgg19",
                  'result':vgg19_CT_predict
               }
            ]
         if model == "3":
            data=[
               {
                  'model':"Inceptionv3",
                  'result':inceptionv3_CT_predict
               }
            ]
         if model == "4":
            data=[
               {
                  'model':"Vgg16",
                  'result':vgg16_CT_predict
               },
               {
                  'model':"Vgg19",
                  'result':vgg19_CT_predict
               },
               {
                  'model':"Inceptionv3",
                  'result':inceptionv3_CT_predict
               }
            ]
         os.remove("./assets/images/upload/"+name)
         print("File removed")
         return render_template('result.html',data=data)



@app.route('/uploader_x', methods = ['POST', 'GET'])
def uploader_x():
   if request.method == 'POST':
      if 'file' not in request.files:
         flash('No file part')
         return render_template('detect.html')
      file = request.files['file']
      name = file.filename
      model = request.form.get("model")  
      if file.filename == '':
         flash('No selected file')
         return render_template('detect.html')
      if model == '0':
         flash('No selected model')
         return render_template('detect.html')
      if file:
         file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
         vgg16 = load_model('models/vgg16/VGG16_CXR_model.h5')
         vgg19 = load_model('models/vgg19/VGG19_CXR_model.h5')
         Resnet50 = load_model('models/resnet50/ResNet50_CXR_model.h5')

         image = cv2.imread("./assets/images/upload/" + name)
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         image = cv2.resize(image, (224, 224))
         image = np.array(image) / 255.0
         image = np.expand_dims(image, axis=0)


         if model == "1" or model == "4":
            vgg16_predict = vgg16.predict(image)
            probability = vgg16_predict[0]
            if probability[1] > 0.5:
               vgg16_x_predict = str('%.2f' % (probability[1] * 100) + '% COVID')
            else:
               vgg16_x_predict = str('%.2f' % ((probability[0]) * 100) + '% NonCOVID')
         
         if model == "2" or model == "4":
            vgg19_predict = vgg19.predict(image)
            probability = vgg19_predict[0]
            if probability[1] > 0.5:
               vgg19_x_predict = str('%.2f' % (probability[1] * 100) + '% COVID')
            else:
               vgg19_x_predict = str('%.2f' % ((probability[0]) * 100) + '% NonCOVID')
         
         if model == "3" or model == "4":
            Resnet50_predict = Resnet50.predict(image)
            probability = Resnet50_predict[0]
            if probability[1] > 0.5:
               Resnet50_x_predict = str('%.2f' % (probability[1] * 100) + '% COVID')
            else:
               Resnet50_x_predict = str('%.2f' % ((probability[0]) * 100) + '% NonCOVID')
         
         if model == "1":
            data=[
               {
                  'model':"Vgg16",
                  'result':vgg16_x_predict
               }
            ]
         if model == "2":
            data=[
               {
                  'model':"Vgg19",
                  'result':vgg19_x_predict
               }
            ]
         if model == "3":
            data=[
               {
                  'model':"Resnet50",
                  'result':Resnet50_x_predict
               }
            ]
         if model == "4":
            data=[
               {
                  'model':"Vgg16",
                  'result':vgg16_x_predict
               },
               {
                  'model':"Vgg19",
                  'result':vgg19_x_predict
               },
               {
                  'model':"Resnet50",
                  'result':Resnet50_x_predict
               }
            ]
         os.remove("./assets/images/upload/"+name)
         print("File removed")
         return render_template('result.html',data=data)


@app.route('/mail', methods = ['POST', 'GET'])
def mail():
   if request.method == "POST":
         name = request.form.get("fullname")
         email = request.form.get("email")
         phone = request.form.get("phone")
         option = request.form.get("option")
         subject = request.form.get("subject")
         message = request.form.get("message")
         m.send_mail(name,email,phone,option,subject,message,"sn.modarressi@aut.ac.ir")
         m.send_mail(name,email,phone,option,subject,message,"maedemir@aut.ac.ir")
   return render_template("contact.html")

if __name__ == '__main__':
   app.secret_key = ".."
   app.run(debug=False)