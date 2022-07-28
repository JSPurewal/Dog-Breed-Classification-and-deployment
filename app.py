import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import joblib
import pickle
import tensorflow as tf
import keras


app=Flask(__name__,static_folder=(r'C:\Users\Jaskaran S. Purewal\Documents\Projects\Image classification\deploy\templates\images'))

#model=joblib.load('models/finalmodel.sav')
model=load_model("models/model1.h5")
#model = load_model('C:/Users/Jaskaran S. Purewal/Documents/Projects/Image classification/deploy/models/model.pkl')

def predict_label(img_path):
    i = keras.utils.load_img(img_path, target_size=(256,256))
    i = np.reshape(i,(-1,256,256,3))
    p = model.predict(i)
    p=p[0]
    m = p[0]
    index = 0
    
    for x in range(1,len(p)):
        if p[x]>m:
            m=p[x]
            index=x
    dict={
        0:'Beagle',
        1:'Bernese Mountain Dog',
        2:'Doberman',
        3:'Labrador',
        4:'Husky'}
    ans=dict[index]
            
    
    return ans

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Made by Jaskaran Singh Purewal"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = img.filename	
        img.save(img_path)
        p = predict_label(img_path)

    return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
    app.run(port=3000,debug = True)