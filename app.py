from __future__ import division, print_function
# coding=utf-8
import cv2
#import tenserflow as tf
import sys
import random
import os
import glob
import re
import pickle
import numpy as np
import aidrtokenize as aidrtokenize
import pandas as pd
import data_process_multimodal_pair as data_process
from crisis_data_generator_image_optimized import DataGenerator 
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.applications.vgg16 import preprocess_input
import data_process_new as dp
from tensorflow.keras.preprocessing import image as im
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from tensorflow import keras
matplotlib.use('Agg')
#tf.compat.v1.disable_eager_execution()
import keras.backend as K
# Define a flask app
app = Flask(__name__)

inf=pd.read_csv("performance_measures/informative.csv")
hum=pd.read_csv("performance_measures/humanitarian.csv")
sev=pd.read_csv("performance_measures/severity.csv")
# Model saved with Keras model.save()
MODEL_PATH1 = 'model/model_info_x.hdf5'
MODEL_PATH2 = 'model/model_info_x1.hdf5'
MODEL_PATH3 = 'model/model_info_x2.hdf5'
MODEL_PATH4 = 'model/model_x.hdf5'
MODEL_PATH5 = 'model/model_x1.hdf5'
MODEL_PATH6 = 'model/model_x2.hdf5'
MODEL_PATH7 = 'model/model_severe_x.hdf5'
MODEL_PATH8 = 'model/model_severe_x1.hdf5'
MODEL_PATH9 = 'model/model_severe_x2.hdf5'
MODEL_PATH10 = 'model/humanitarian_image_vgg16_ferda.hdf5'
MODEL_PATH11 = 'model/humanitarian_cnn_keras_09-04-2022_05-10-03.hdf5'
MODEL_PATH12 = 'model/informativeness_cnn_keras.hdf5'
MODEL_PATH13 = 'model/informative_image.hdf5'
MODEL_PATH14 = 'model/severity_cnn_keras_21-07-2022_08-14-32.hdf5'
MODEL_PATH15 = 'model/severity_image.hdf5'

with open("data_dump/all_images_data_dump.npy", 'rb') as handle:                              
      images_npy_data = pickle.load(handle)

#dummy_images = np.random.randint(0, 256, (10, 256, 256, 4), dtype=np.uint8)
#images_npy_data =  dummy_images 

# Load your trained model
path="metadata/task_informative_text_img_agreed_lab_test.tsv"
path1="metadata/task_humanitarian_text_img_agreed_lab_test.tsv"
path2="metadata/task_severity_test.tsv"
df=pd.read_csv(path,sep="\t")
df1=pd.read_csv(path1,sep="\t")
df2=pd.read_csv(path2,sep="\t")
img1=list(df['image'].values)
img2=list(df1['image'].values)
img3=list(df2['image'].values)
text1=list(df['tweet_text'].values)
text2=list(df1['tweet_text'].values)
text3=list(df2['tweet_text'].values)
label1=list(df['label'].values)

df1.loc[df1["label"] == "missing_or_found_people", "label"] = "affected_individuals"
df1.loc[df1["label"] == "injured_or_dead_people", "label"] = "affected_individuals"
df1.loc[df1["label"] == "vehicle_damage", "label"] = "infrastructure_and_utility_damage"
label2=list(df1['label'].values)
label3=list(df2['label'].values)
random_index=0
model1 = load_model(MODEL_PATH1)
model2 = load_model(MODEL_PATH2)
model3 = load_model(MODEL_PATH3)
model4 = load_model(MODEL_PATH4)
model5 = load_model(MODEL_PATH5)
model6 = load_model(MODEL_PATH6)
model7 = load_model(MODEL_PATH7)
model8 = load_model(MODEL_PATH8)
model9 = load_model(MODEL_PATH9)
model10 = load_model(MODEL_PATH10)
model11 = load_model(MODEL_PATH11)
model12 = load_model(MODEL_PATH12)
model13 = load_model(MODEL_PATH13)
model14 = load_model(MODEL_PATH14)
model15 = load_model(MODEL_PATH15)
#model1._make_predict_function()          # Necessary
#model2._make_predict_function()
#model3._make_predict_function()
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')




def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    Returns a numpy 3darray (the preprocessed image).
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

def save_image(image,hm):


    img = keras.preprocessing.image.load_img(image)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    hm = np.uint8(255 * hm)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[hm]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    plt.clf()
    plt.matshow(superimposed_img)
    plt.colorbar()
    # Save the image to disk
    plt.savefig('./static/visualize.jpg')

def _get_text_xticks(sentence):
    tokens = [word_.strip() for word_ in sentence.split(' ')]
    return tokens 

def _plot_score(vec, pred_text, xticks):
    #_axis_fontsize=60
    print(vec)
    plt.clf()
    fig=plt.figure(figsize = (5,4))
    plt.yticks([])
    plt.xticks(range(0,len(vec)), xticks, fontsize= 15,rotation='vertical')
    #fig.add_subplot(1, 1, 1)
    #plt.figtext(x=0.13, y=0.54, s='Prediction: {}'.format(pred_text), fontsize=15, fontname='sans-serif')
    #plt.subplots_adjust(bottom=5.0,top=5.1)
    img = plt.imshow([vec], vmin=0, vmax=1,origin="lower")
    plt.colorbar()
    plt.savefig('./static/text.jpg')

    #plt.show()    

def data_generation(image_file_list,text_x,n_classes,images_npy_data,labels,max_seq_length):
        # Initialization
        y = np.empty((len(text_x),n_classes), dtype=int)
        text_batch = np.empty((len(text_x), max_seq_length), dtype=int)
        images_batch = np.empty([len(text_x), 224, 224, 3])
        indexes = np.arange(len(image_file_list))
        # Generate data
        for i, index in enumerate(indexes):
            #print(index)
                if(index <= len(image_file_list)):
                    # if (index in self.image_file_list):
                    image_file_name = str(image_file_list[index])
                    #print(image_file_name)
                    if(image_file_name in images_npy_data):
                        # if(image_file_name=="image_null"):
                        #     img = np.zeros([1, 224, 224, 3])
                        # else:
                        img = images_npy_data[image_file_name]
                        #img = image.load_img(self.image_file_list[index], target_size=(224, 224))
                        #img = image.img_to_array(img)
                        #img = np.expand_dims(img, axis=0)
                        #img = preprocess_input(img)
                        images_batch[i, :, :, :] = img
                        # Store class
                        #y[i] = labels[index]
                        text_batch[i] = text_x[index]

        current_images_batch = preprocess_input_vgg(images_batch)
        return current_images_batch, text_batch, labels

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET','POST'])
def index():

    # Main page
    #print(img1[random_index])
    random_index1 = random.randint(0,len(img1)-1)
    random_index2 = random.randint(0,len(img2)-1)
    random_index3 = random.randint(0,len(img3)-1)
    index=min(min(random_index1,random_index2),random_index3)
    print(img1[index:index+4])
    print(img2[index:index+4])
    print(img3[index:index+4])
    #checked=0
    return render_template('index.html',img1=img1,img2=img2,img3=img3,text1=text1,text2=text2,text3=text3,radio=1,m1={},m2={},m3={},l1=0,l2=0,l3=0,index=index,output=None,result={},len=0,labels=[],i="0")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        print(request.form['index'])
        if(request.form['inlineRadioOptions']=="option1"):
            print("Enter")  
            image_file_list=img1[int(request.form['index']):int(request.form['index'])+4]
            for i in range(len(image_file_list)):
                image_file_list[i]=image_file_list[i]
            text_file_list=text1[int(request.form['index']):int(request.form['index'])+4]
            label=label1[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list,text_file_list,label)
            
            tokenizer = pickle.load( open( "model/info_multimodal_paired_agreed_lab.tokenizer", "rb" ))
            
            test_x, test_image_list, test_y, test_le, test_labels = data_process.read_dev_data_multimodal(image_file_list,text_file_list,label,tokenizer,25,"\t",label1)
            print(test_x, test_image_list, test_y, test_le)
            params = {"max_seq_length": 25, "batch_size": 4, "n_classes": 2, "shuffle": False} 
            test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)
            #n_classes=2
            output1=model1.predict(test_data_generator,verbose=1)
            output2=model2.predict(test_data_generator,verbose=1)
            output3=model3.predict(test_data_generator,verbose=1)
            preds=[output1,output2,output3]
            preds = np.array(preds)                                                     
            summed = np.sum(preds,axis=0)                                               
            a=np.argmax(summed,axis=1)
            print(summed)
            print(test_labels)
            output=[test_labels[a[0]],test_labels[a[1]],test_labels[a[2]],test_labels[a[3]]]
            result=dict()
            result1=dict()
            result2=dict()
            result3=dict()
            for i in range(len(test_labels)):
                result[test_labels[i]]=summed[0][i]
                result1[test_labels[i]]=summed[1][i]
                result2[test_labels[i]]=summed[2][i]
                result3[test_labels[i]]=summed[3][i]
            print(test_labels)
            if(int(request.form['index1'])==0):
                result=result
            elif(int(request.form['index1'])==1):
                result=result1
            elif(int(request.form['index1'])==2):
                result=result2
            else:
                result=result3  
            
            m1=dict()
            m2=dict()
            m3=dict()
            for i in range(len(test_labels)):
                m1[test_labels[i]]=output1[int(request.form['index1'])][i]
                m2[test_labels[i]]=output2[int(request.form['index1'])][i]
                m3[test_labels[i]]=output3[int(request.form['index1'])][i]
                #result3[test_labels[i]]=summed[3][i]
            #result1=dict()
            #ind=0
            #while(ind<len(output1[0])):
            #     res
            #a,b,c=data_generation(test_image_list,test_x,n_classes,images_npy_data,test_y,25)
            #print(a,b,c)
            return render_template('index.html',img1=img1,img2=img2,img3=img3,text1=text1,text2=text2,text3=text3,radio=1,m1=m1,m2=m2,m3=m3,l1=len(m1.keys()),l2=len(m2.keys()),l3=len(m3.keys()),index=int(request.form['index']),output=output,result=result,len=len(result.keys()),labels=test_labels,i=request.form['index1'])

        if(request.form['inlineRadioOptions']=="option2"):
            #print("Enter")  
            image_file_list=img2[int(request.form['index']):int(request.form['index'])+4]
            for i in range(len(image_file_list)):
                image_file_list[i]=image_file_list[i]
            text_file_list=text2[int(request.form['index']):int(request.form['index'])+4]
            print(set(label2))
            label=label2[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list,text_file_list,label)
            
            tokenizer = pickle.load( open( "model/info_multimodal_paired_agreed_lab.tokenizer", "rb" ))
            #with open("all_images_data_dump.npy", 'rb') as handle:                              
            #       images_npy_data = pickle.load(handle) 
            test_x, test_image_list, test_y, test_le, test_labels = data_process.read_dev_data_multimodal(image_file_list,text_file_list,label,tokenizer,25,"\t",label2)
            print(test_x, test_image_list, test_y, test_le)
            params = {"max_seq_length": 25, "batch_size": 4, "n_classes": 5, "shuffle": False} 
            test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)
            #n_classes=2
            output1=model4.predict(test_data_generator,verbose=1)
            output2=model5.predict(test_data_generator,verbose=1)
            output3=model6.predict(test_data_generator,verbose=1)
            preds=[output1,output2,output3]
            preds = np.array(preds)                                                     
            summed = np.sum(preds,axis=0)                                               
            a=np.argmax(summed,axis=1)
            output=[test_labels[a[0]],test_labels[a[1]],test_labels[a[2]],test_labels[a[3]]]
            print(summed)
            result=dict()
            result1=dict()
            result2=dict()
            result3=dict()
            for i in range(len(test_labels)):
                result[test_labels[i]]=summed[0][i]
                result1[test_labels[i]]=summed[1][i]
                result2[test_labels[i]]=summed[2][i]
                result3[test_labels[i]]=summed[3][i]
            print(test_labels)

            if(int(request.form['index1'])==0):
                result=result
            elif(int(request.form['index1'])==1):
                result=result1
            elif(int(request.form['index1'])==2):
                result=result2
            else:
                result=result3    

            m1=dict()
            m2=dict()
            m3=dict()
            for i in range(len(test_labels)):
                m1[test_labels[i]]=output1[int(request.form['index1'])][i]
                m2[test_labels[i]]=output2[int(request.form['index1'])][i]
                m3[test_labels[i]]=output3[int(request.form['index1'])][i]            
            #result1=dict()
            #ind=0
            #while(ind<len(output1[0])):
            #     res
            #a,b,c=data_generation(test_image_list,test_x,n_classes,images_npy_data,test_y,25)
            #print(a,b,c)
            return render_template('index.html',img1=img2,img2=img1,img3=img3,text1=text2,text2=text1,text3=text3,radio=2,m1=m1,m2=m2,m3=m3,l1=len(m1.keys()),l2=len(m2.keys()),l3=len(m3.keys()),index=int(request.form['index']),output=output,result=result,len=len(result.keys()),labels=test_labels,i=request.form['index1'])    
        

        if(request.form['inlineRadioOptions']=="option3"):
            #print("Enter")  
            image_file_list=img3[int(request.form['index']):int(request.form['index'])+4]
            for i in range(len(image_file_list)):
                image_file_list[i]=image_file_list[i]
            text_file_list=text3[int(request.form['index']):int(request.form['index'])+4]
            print(set(label2))
            label=label3[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list,text_file_list,label)
            
            tokenizer = pickle.load( open( "model/info_multimodal_paired_agreed_lab.tokenizer", "rb" ))
            #with open("all_images_data_dump.npy", 'rb') as handle:                              
            #       images_npy_data = pickle.load(handle) 
            test_x, test_image_list, test_y, test_le, test_labels = data_process.read_dev_data_multimodal(image_file_list,text_file_list,label,tokenizer,25,"\t",label3)
            print(test_x, test_image_list, test_y, test_le)
            params = {"max_seq_length": 25, "batch_size": 4, "n_classes": 3, "shuffle": False} 
            test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)
            #n_classes=2
            output1=model7.predict(test_data_generator,verbose=1)
            output2=model8.predict(test_data_generator,verbose=1)
            output3=model9.predict(test_data_generator,verbose=1)
            preds=[output1,output2,output3]
            preds = np.array(preds)                                                     
            summed = np.sum(preds,axis=0)                                               
            a=np.argmax(summed,axis=1)
            output=[test_labels[a[0]],test_labels[a[1]],test_labels[a[2]],test_labels[a[3]]]
            print(summed)
            result=dict()
            result1=dict()
            result2=dict()
            result3=dict()
            for i in range(len(test_labels)):
                result[test_labels[i]]=summed[0][i]
                result1[test_labels[i]]=summed[1][i]
                result2[test_labels[i]]=summed[2][i]
                result3[test_labels[i]]=summed[3][i]
            print(test_labels)

            if(int(request.form['index1'])==0):
                result=result
            elif(int(request.form['index1'])==1):
                result=result1
            elif(int(request.form['index1'])==2):
                result=result2
            else:
                result=result3    

            m1=dict()
            m2=dict()
            m3=dict()
            for i in range(len(test_labels)):
                m1[test_labels[i]]=output1[int(request.form['index1'])][i]
                m2[test_labels[i]]=output2[int(request.form['index1'])][i]
                m3[test_labels[i]]=output3[int(request.form['index1'])][i]            
            #result1=dict()
            #ind=0
            #while(ind<len(output1[0])):
            #     res
            #a,b,c=data_generation(test_image_list,test_x,n_classes,images_npy_data,test_y,25)
            #print(a,b,c)
            return render_template('index.html',img1=img3,img2=img2,img3=img1,text1=text3,text2=text2,text3=text1,radio=3,m1=m1,m2=m2,m3=m3,l1=len(m1.keys()),l2=len(m2.keys()),l3=len(m3.keys()),index=int(request.form['index']),output=output,result=result,len=len(result.keys()),labels=test_labels,i=request.form['index1'])
        #print(a.shape,b.shape,c.shape)
        #a = [a[0],b[0]]
        #c = c[0]
        #model1.predict(a,c,verbose=1)
        #model2.predict(a,c,verbose=1)
        #model3.predict(a,c,verbose=1)
        #model2.predict_generator(test_data_generator, verbose=1)
        #model3.predict_generator(test_data_generator, verbose=1)
        # Make prediction
        #preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return "Hey"
    return None

@app.route('/details', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        print(request.form['index2'])
        if(request.form['inlineRadioOptions']=="option1"):
            #print("Enter")
            #print(type(request.form['index1'])) 
            a=re.search(r"\{(.*?)\}", request.form['index2'])
            a = eval(a.group(0))
            print(a,type(a))
            val1=dict()
            for key,val in a.items():
                val1[key]=val
            image_file_list=img1[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list) 
            for i in range(len(image_file_list)):
                image_file_list[i]=image_file_list[i]
            text_file_list=text1[int(request.form['index']):int(request.form['index'])+4]
            label=label1[int(request.form['index']):int(request.form['index'])+4]
            #print(text_file_list,label)
            
            tokenizer = pickle.load( open( "model/informativeness_cnn_keras_09-04-2022_04-26-49.tokenizer", "rb" ))
            
            test_x, test_y, test_le, test_labels = dp.read_dev_data(text_file_list,label,tokenizer,25,"\t",label1)
            test_data,test_label,_,_ = dp.generate_data_file(image_file_list,label,"\t",label1)
            test_data = preprocess_input_vgg(test_data)
            #print(test_data)
            #print(test_x, test_image_list, test_y, test_le)
            #params = {"max_seq_length": 25, "batch_size": 4, "n_classes": 2, "shuffle": False} 
            #test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)
            #n_classes=2
            result=dict()
            result1=dict()
            #print(test_x)
            output1=model12.predict((test_x), batch_size=128, verbose=1)
            output2=model13.predict([test_data], batch_size=128, verbose=1)
            for i in range(len(test_labels)):
                result[test_labels[i]]=output1[int(request.form['index1'])][i]
                result1[test_labels[i]]=output2[int(request.form['index1'])][i]
            #print(result)
            a=np.argmax(output1[int(request.form['index1'])])
            a1=np.argmax(output2[int(request.form['index1'])])
            print(a)    
            output=test_labels[a] 
            output1=test_labels[a1]
            index=int(request.form['index1'])
            print(image_file_list[index])   
            #print()    
            return render_template('result.html',m3=val1,output2=request.form['index3'],index=index,img=image_file_list,text=text_file_list,m1=result,m2=result1,len=0,labels=test_labels,output1=output1,output=output,l1=len(result.keys()),l2=len(result1.keys()),l3=len(val1.keys()),column_names=inf.columns.values, row_data=list(inf.values.tolist()),model_name="Informative")
        if(request.form['inlineRadioOptions']=="option2"):
            #print("Enter")  
            #print(image_file_list)
            #print(type(request.form['index1'])) 
            a=re.search(r"\{(.*?)\}", request.form['index2'])
            
            a = eval(a.group(0))
            print(a,type(a))
            val1=dict()
            for key,val in a.items():
                val1[key]=val
            image_file_list=img2[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list)
            for i in range(len(image_file_list)):
                image_file_list[i]=image_file_list[i]
            text_file_list=text2[int(request.form['index']):int(request.form['index'])+4]
            #print(set(label2))
            label=label2[int(request.form['index']):int(request.form['index'])+4]
            #print(image_file_list,text_file_list,label)
            
            tokenizer = pickle.load( open( "model/humanitarian_cnn_keras_09-04-2022_05-10-03.tokenizer", "rb" ))
            #with open("all_images_data_dump.npy", 'rb') as handle:                              
            #       images_npy_data = pickle.load(handle) 
            test_x, test_y, test_le, test_labels = dp.read_dev_data(text_file_list,label,tokenizer,25,"\t",label2)
            test_data,test_label,_,_ = dp.generate_data_file(image_file_list,label,"\t",label2)
            test_data = preprocess_input_vgg(test_data)
            #print(test_x, test_image_list, test_y, test_le)
            #params = {"max_seq_length": 25, "batch_size": 4, "n_classes": 2, "shuffle": False} 
            #test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)
            #n_classes=2
            result=dict()
            result1=dict()
            #print(test_x)
            output1=model11.predict((test_x), batch_size=128, verbose=1)
            output2=model10.predict((test_data), batch_size=128, verbose=1)
            for i in range(len(test_labels)):
                result[test_labels[i]]=output1[int(request.form['index1'])][i]
                result1[test_labels[i]]=output2[int(request.form['index1'])][i]
            #print(result,result1)
            #print(test_labels)
            a=np.argmax(output1[int(request.form['index1'])])
            a1=np.argmax(output2[int(request.form['index1'])])
            #print(a,a1)    
            output=test_labels[a] 
            output1=max(result1, key=result1.get)
            index=int(request.form['index1'])   
            print(image_file_list[index])     
            return render_template('result.html',m3=val1,output2=request.form['index3'],index=index,img=image_file_list,text=text_file_list,m1=result,m2=result1,len=0,labels=test_labels,output1=output1,output=output,l1=len(result.keys()),l2=len(result1.keys()),l3=len(val1.keys()),column_names=hum.columns.values, row_data=list(hum.values.tolist()),model_name="Damage")

        if(request.form['inlineRadioOptions']=="option3"):
            #print("Enter")  
            a=re.search(r"\{(.*?)\}", request.form['index2'])
            a = eval(a.group(0))
            print(a,type(a))
            val1=dict()
            for key,val in a.items():
                val1[key]=val
            image_file_list=img3[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list)
            for i in range(len(image_file_list)):
                image_file_list[i]=image_file_list[i]
            text_file_list=text3[int(request.form['index']):int(request.form['index'])+4]
            print(set(label2))
            label=label3[int(request.form['index']):int(request.form['index'])+4]
            print(image_file_list,text_file_list,label)
            
            tokenizer = pickle.load( open( "model/severity_cnn_keras_21-07-2022_08-14-32.tokenizer", "rb" ))
            #with open("all_images_data_dump.npy", 'rb') as handle:                              
            #       images_npy_data = pickle.load(handle) 
            test_x, test_y, test_le, test_labels = dp.read_dev_data(text_file_list,label,tokenizer,25,"\t",label3)
            test_data,test_label,_,_ = dp.generate_data_file(image_file_list,label,"\t",label3)
            test_data = preprocess_input_vgg(test_data)
            #print(test_x, test_image_list, test_y, test_le)
            #params = {"max_seq_length": 25, "batch_size": 4, "n_classes": 2, "shuffle": False} 
            #test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)
            #n_classes=2
            result=dict()
            result1=dict()
            print(test_x)
            output1=model14.predict((test_x), batch_size=128, verbose=1)
            output2=model15.predict((test_data), batch_size=128, verbose=1)
            for i in range(len(test_labels)):
                result[test_labels[i]]=output1[int(request.form['index1'])][i]
                result1[test_labels[i]]=output2[int(request.form['index1'])][i]
            print(result,result1)
            #print(test_labels)
            a=np.argmax(output1[int(request.form['index1'])])
            a1=np.argmax(output2[int(request.form['index1'])])
            #print(a,a1)    
            output=test_labels[a] 
            output1=max(result1, key=result1.get)
            index=int(request.form['index1'])
            #print(image_file_list[index])   
            print(val1,result,result1)    
            return render_template('result.html',m3=val1,output2=request.form['index3'],index=index,img=image_file_list,text=text_file_list,m1=result,m2=result1,len=0,labels=test_labels,output1=output1,output=output,l1=len(result.keys()),l2=len(result1.keys()),l3=len(val1.keys()),column_names=sev.columns.values, row_data=list(sev.values.tolist()),model_name="Severity")

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():         
    if request.method == 'POST':
        print(request.form['index2'])
        if(request.form['inlineRadioOptions']=="option1"):
            index=int(request.form['index1'])+int(request.form['index'])
            image=img1[index]
            text=text1[index]
            label=label1[index]
            pred_vector_output = model1.output[:, 0]
            print(pred_vector_output)
            img = im.load_img(image, target_size=(224, 224))

            # `x` is a float32 Numpy array of shape (224, 224, 3)
            x = im.img_to_array(img)

            # We add a dimension to transform our array into a "batch"
            # of size (1, 224, 224, 3)
            x = np.expand_dims(x, axis=0)

            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            x = preprocess_input(x)
            txt = aidrtokenize.tokenize(text)
            print(txt)
            
            tokenizer = pickle.load( open( "model/info_multimodal_paired_agreed_lab.tokenizer", "rb" ))
            sequences = tokenizer.texts_to_sequences(txt)
            print(sequences)
            data = pad_sequences(sequences, maxlen=25, padding='post')
            
            #tf.compat.v1.disable_eager_execution()
            
            #K.set_image_data_format('channels_last')
            #tf.compat.v1.disable_eager_execution()
            
            conv_layer = model1.get_layer("block5_conv3")
            heatmap_model = Model([model1.inputs], [conv_layer.output, model1.output])

            # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model([x,data])
                loss = predictions[:, np.argmax(predictions[0])]
                grads = gtape.gradient(loss, conv_output)
                pooled_grads = K.mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

            print(heatmap.shape)
            heatmap=np.squeeze(heatmap)
            print(heatmap)
            # fig=plt.figure()
            # ax=fig.add_subplot(1,1,1)
            # plot=ax.pcolor(heatmap)
            # fig.colorbar(plot)
            # fig.savefig("./static/heatmap.jpg")
            save_image(image,heatmap)
            conv_layer = model1.get_layer("concatenate")
            heatmap_model = Model([model1.inputs], [conv_layer.output, model1.output])
            # This is the gradient of the predicted class with regard to
            # the output feature map of selected block
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model([x,data])
                loss = predictions[:, np.argmax(predictions[0])]
                grads = gtape.gradient(loss, conv_output)
                grads /= (np.max(grads) + K.epsilon())
                pooled_grads = K.mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat
            _grad_CAM=tf.squeeze(heatmap)
            print(_grad_CAM.shape)
            arr_min, arr_max= np.min(_grad_CAM), np.max(_grad_CAM)
            print(arr_min, arr_max)
            grad_CAM= (_grad_CAM - arr_min) / (arr_max - arr_min + K.epsilon())
            print(_grad_CAM)
            _plot_score(vec=_grad_CAM[:len(_get_text_xticks(txt))], pred_text="Informative", xticks=_get_text_xticks(txt))
            return render_template('visualize.html',image=image,text=text,img1=img1,img2=img2,img3=img3,text1=text1,text2=text2,text3=text3,radio=1,m1={},m2={},m3={},l1=0,l2=0,l3=0,index=index,output=None,result={},len=0,labels=[],i="0")
            #print()    
            #return render_template('result.html',m2=val1,output2=request.form['index3'],index=index,img=image_file_list,text=text_file_list,result=result,m1=result1,len=len(result.keys()),labels=test_labels,output1=output1,output=output,l1=0,l2=0,l3=0)

        if(request.form['inlineRadioOptions']=="option2"):
            #print("Enter")  
            #print(image_file_list)
            #print(type(request.form['index1'])) 
            index=int(request.form['index1'])+int(request.form['index'])
            image=img2[index]
            text=text2[index]
            label=label2[index]
            pred_vector_output = model1.output[:, 0]
            print(pred_vector_output)
            img = im.load_img(image, target_size=(224, 224))

            # `x` is a float32 Numpy array of shape (224, 224, 3)
            x = im.img_to_array(img)

            # We add a dimension to transform our array into a "batch"
            # of size (1, 224, 224, 3)
            x = np.expand_dims(x, axis=0)

            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            x = preprocess_input(x)
            txt = aidrtokenize.tokenize(text)
            print(txt)
            
            tokenizer = pickle.load( open( "model/info_multimodal_paired_agreed_lab.tokenizer", "rb" ))
            sequences = tokenizer.texts_to_sequences(txt)
            print(sequences)
            data = pad_sequences(sequences, maxlen=25, padding='post')
            
            #tf.compat.v1.disable_eager_execution()
            
            #K.set_image_data_format('channels_last')
            #tf.compat.v1.disable_eager_execution()
            
            conv_layer = model1.get_layer("block5_conv3")
            heatmap_model = Model([model1.inputs], [conv_layer.output, model1.output])

            # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model([x,data])
                loss = predictions[:, np.argmax(predictions[0])]
                grads = gtape.gradient(loss, conv_output)
                pooled_grads = K.mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

            print(heatmap.shape)
            heatmap=np.squeeze(heatmap)
            print(heatmap)
            
            save_image(image,heatmap)
            conv_layer = model1.get_layer("concatenate")
            heatmap_model = Model([model1.inputs], [conv_layer.output, model1.output])
            # This is the gradient of the predicted class with regard to
            # the output feature map of selected block
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model([x,data])
                loss = predictions[:, np.argmax(predictions[0])]
                grads = gtape.gradient(loss, conv_output)
                grads /= (np.max(grads) + K.epsilon())
                pooled_grads = K.mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat
            _grad_CAM=tf.squeeze(heatmap)
            print(_grad_CAM.shape)
            arr_min, arr_max= np.min(_grad_CAM), np.max(_grad_CAM)
            print(arr_min, arr_max)
            grad_CAM= (_grad_CAM - arr_min) / (arr_max - arr_min + K.epsilon())
            print(_grad_CAM)
            _plot_score(vec=_grad_CAM[:len(_get_text_xticks(txt))], pred_text="Informative", xticks=_get_text_xticks(txt))     
            return render_template('visualize.html',image=image,text=text,img1=img1,img2=img2,img3=img3,text1=text1,text2=text2,text3=text3,radio=1,m1={},m2={},m3={},l1=0,l2=0,l3=0,index=index,output=None,result={},len=0,labels=[],i="0")  
            #return render_template('result.html',m2=val1,output2=request.form['index3'],index=index,img=image_file_list,text=text_file_list,result=result,m1=result1,len=len(result.keys()),labels=test_labels,output1=output1,output=output,l1=0,l2=0,l3=0)

        if(request.form['inlineRadioOptions']=="option3"):
            #print("Enter")  
            index=int(request.form['index1'])+int(request.form['index'])
            image=img3[index]
            text=text3[index]
            label=label3[index]
            pred_vector_output = model7.output[:, 0]
            print(pred_vector_output)
            img = im.load_img(image, target_size=(224, 224))

            # `x` is a float32 Numpy array of shape (224, 224, 3)
            x = im.img_to_array(img)

            # We add a dimension to transform our array into a "batch"
            # of size (1, 224, 224, 3)
            x = np.expand_dims(x, axis=0)

            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            x = preprocess_input(x)
            txt = aidrtokenize.tokenize(text)
            print(txt)
            
            tokenizer = pickle.load( open( "model/info_multimodal_paired_agreed_lab.tokenizer", "rb" ))
            sequences = tokenizer.texts_to_sequences(txt)
            print(sequences)
            data = pad_sequences(sequences, maxlen=25, padding='post')
            
            #tf.compat.v1.disable_eager_execution()
            
            #K.set_image_data_format('channels_last')
            #tf.compat.v1.disable_eager_execution()
            
            conv_layer = model7.get_layer("block5_conv3")
            heatmap_model = Model([model7.inputs], [conv_layer.output, model7.output])

            # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model([x,data])
                loss = predictions[:, np.argmax(predictions[0])]
                grads = gtape.gradient(loss, conv_output)
                pooled_grads = K.mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

            print(heatmap.shape)
            heatmap=np.squeeze(heatmap)
            print(heatmap)
            # fig=plt.figure()
            # ax=fig.add_subplot(1,1,1)
            # plot=ax.pcolor(heatmap)
            # fig.colorbar(plot)
            # fig.savefig("./static/heatmap.jpg")
            save_image(image,heatmap)
            conv_layer = model7.get_layer("concatenate")
            heatmap_model = Model([model7.inputs], [conv_layer.output, model7.output])
            # This is the gradient of the predicted class with regard to
            # the output feature map of selected block
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model([x,data])
                loss = predictions[:, np.argmax(predictions[0])]
                grads = gtape.gradient(loss, conv_output)
                grads /= (np.max(grads) + K.epsilon())
                pooled_grads = K.mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat
            _grad_CAM=tf.squeeze(heatmap)
            print(_grad_CAM.shape)
            arr_min, arr_max= np.min(_grad_CAM), np.max(_grad_CAM)
            print(arr_min, arr_max)
            grad_CAM= (_grad_CAM - arr_min) / (arr_max - arr_min + K.epsilon())
            print(_grad_CAM)
            _plot_score(vec=_grad_CAM[:len(_get_text_xticks(txt))], pred_text="Informative", xticks=_get_text_xticks(txt))   
            #print()    
            return render_template('visualize.html',image=image,text=text,img1=img1,img2=img2,img3=img3,text1=text1,text2=text2,text3=text3,radio=1,m1={},m2={},m3={},l1=0,l2=0,l3=0,index=index,output=None,result={},len=0,labels=[],i="0")    


if __name__ == '__main__':
    app.run()
