import streamlit as st
import io
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import cv2
import time
import os
import glob
import scipy.spatial.distance as distance
import re

import torch
import tensorflow as tf
from keras import Model


target_classess = ["Bed", "Cabinetry", "Chair", "Couch", "Lamp", "Table"]
app_dir =  'appdata/'
save_img_dir = "./appdata/detected_images/detect/"
dfpath = 'ikeadata/ikea_final_model0.csv'


@st.cache_resource
def load_ikeadf(path):
    df = pd.read_csv(path)
    vector = df['vector'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
    return vector, df

#Initialize the detectron model
@st.cache_resource
def initialization():
    predictor = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/object_detection_model/yolov5m_weights.pt')
    return predictor


@st.cache_resource
def load_similarity_model():
    model = tf.keras.models.load_model("./models/similarity_model/multilabel0")
    model_new = Model(model.input, model.get_layer('dense_4').output)
    return model_new


def preprocess_for_predict(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(100,100))
    imgarr = img.reshape((1,) + img.shape)
    return imgarr

def getpred(img,pred_model):
    preds = pred_model.predict(img)
    return preds

def calculate_similarity(vector1,vector2):
    return 1-distance.cosine(vector1, vector2)


def compare_similarity(image_arr,model_dense):
    vector_obj = getpred(image_arr,model_dense)
    ikeadf['similarity'] = vector.apply(lambda x:calculate_similarity(x,vector_obj))
    return ikeadf.sort_values(by=['similarity'],ascending=False).head(1000)

def clearold():
    files = glob.glob('appdata/'+'*.*g')
    if files:
        for f in files:
            os.remove(f)
            
#app start
# initializing yolov5 model for object detection
st.set_page_config(layout="wide")
predictor = initialization()
predictor = predictor.cpu()

# Setting the confidence threshold (higher the threshold lower the FPs) configure accordingly
predictor.conf = 0.45

# initializing vgg16 model for feature extraction
model = load_similarity_model()


# st.image(Image.open("idecor_logo.png"), width = 700)
st.write('**_Here is where you furnish your home with a Click, just from your couch._**')
st.sidebar.header("Choose a furniture image for recommendation.")


clearold()

uploaded_file = st.file_uploader("Choose an image with .jpg format", type="jpg")


if uploaded_file is not None:
    #save user image and display success message
    vector, ikeadf = load_ikeadf(dfpath)
    
    shutil.rmtree(save_img_dir)
    image = Image.open(uploaded_file)
    user_img_path = app_dir+uploaded_file.name
    image.save(user_img_path)
    
    st.sidebar.image(image,width = 250)

    st.sidebar.success('Upload Successful! Please wait for object detection.')

    #get image list from the detectron model
    with st.spinner('Working hard on finding furniture...'):
        #delete unncessay history file
        clearold()

        detected_objs = predictor(image).crop(save_dir= save_img_dir)
        
        furniturelist = np.array([])
        label_list = np.array([])
        furniture_dict = {}
        

        for obj in detected_objs:
            label = obj['label'][:-5]
            if label not in label_list:
                label_list = np.append(label_list, [label])
                furniture_dict[label] = glob.glob(f"appdata/detected_images/detect/crops/{label}/*.jp*")

        #open cropped image of furniture
        for furniture, image_list in furniture_dict.items():
            for i, image_path in enumerate(image_list):
                st.sidebar.write(f"{furniture}-{i+1}")
                furniturelist = np.append(furniturelist, [f"{furniture}-{i+1}"])
                st.sidebar.image(Image.open(image_path),width = 150)
                
        furniturelist.sort()
    #provide select box for selection
    display = furniturelist
    options = list(range(len(furniturelist)))
    option = st.selectbox('Which furniture do you want to look for?', options, format_func=lambda x: display[x])
    
    if len(detected_objs) == 0:
        st.warning('No objects detected, please re-upload the image.', icon="⚠️")

    else:
        if st.button('Confirm to select '+furniturelist[option]):

            furniture_type = display[option].split('-')[0]
            image_index = int(display[option].split('-')[1]) - 1

            pred_path = furniture_dict[furniture_type][image_index]
            image_array = preprocess_for_predict(pred_path)
            df = compare_similarity(image_array, model)
            obj_class = furniture_type.lower() 
            st.write("Recommendation for: "+ obj_class)

            c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1, 1))
            columnli = [c1,c2,c3,c4,c5]

            for i,column in enumerate(columnli):
                coltitle = re.match(r"^([^,]*)",str(df[df['item_cat']==obj_class][i:i+1].item_name.values.astype(str)[0])).group()
                colcat = str(df[df['item_cat']==obj_class][i:i+1].item_cat.values.astype(str)[0])
                colpic = str(df[df['item_cat']==obj_class][i:i+1].index.values.astype(str)[0])
                colprice = '$' + str(df[df['item_cat']==obj_class][i:i+1].item_price.values.astype(str)[0])
                collink = str(df[df['item_cat']==obj_class][i:i+1].prod_url.values.astype(str)[0])
                colurl = 'ikeadata/'+colcat+'/'+colpic+'.jpg'
                column.image(Image.open(colurl),width=180)
                column.write('### '+colprice)  
                column.write('##### '+coltitle)
                column.write("##### "+"[View more product info]("+collink+")")
                # column.write("[!["+coltitle+"]("+colurl+")]("+collink+")")

            st.text("")
            st.write("Some other non-"+furniture_type+"s items you may like: ")

            c6,c7,c8,c9,c10 = st.columns((1, 1, 1, 1, 1))
            columnli2 = [ c6,c7,c8,c9,c10]
            for i,column in enumerate(columnli2):
                coltitle = re.match(r"^([^,]*)",str(df[df['item_cat']!=obj_class][i:i+1].item_name.values.astype(str)[0])).group()
                colcat = str(df[df['item_cat']!=obj_class][i:i+1].item_cat.values.astype(str)[0])
                colpic = str(df[df['item_cat']!=obj_class][i:i+1].index.values.astype(str)[0])
                colprice = '$' + str(df[df['item_cat']!=obj_class][i:i+1].item_price.values.astype(str)[0])
                colurl = 'ikeadata/'+colcat+'/'+colpic+'.jpg'
                column.image(Image.open(colurl),width=180)
                column.write('### '+colprice)  
                column.write('##### '+coltitle)
                column.write("##### "+"[View more product info]("+collink+")")
