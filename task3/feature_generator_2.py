import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import pickle
from zipfile import ZipFile
from google.colab import drive

#############
#GENERAL SETTINGS REGARDING PICS QUALITY
target_shape = [299, 299]
input_shape = (299, 299, 3)

seed = 7
random.seed(seed)
tf.random.set_seed(seed)#dont know if it helps

#############
def generator_pictures(pics_dir, ids_dict):
    #https://www.geeksforgeeks.org/generators-in-python/
    i = 0
    while True:
        batch_lst = []
        # for i in range(len(pics_lst[0:num_images])):
        while len(batch_lst) < 1:
            #create input array with onyl 1 input from image, iteratively then all other images input are loaded
            pic_id = ids_dict[i]
            pic = load_img(pics_dir + '/' + pic_id)
            # pic = load_img(pics_dir + '\\' + pic_id)
            pic_ary = img_to_array(pic)
            # not all pictures have same shape, we need thus to reshape them
            pic_mod = tf.image.resize_with_pad(pic_ary, target_shape[0], target_shape[1], antialias=True)
            pic_pic = array_to_img(pic_mod)
            #converting back and forth pictures produced a better working features
            #tf.image.decode_jpeg
            pic = tf.keras.applications.inception_resnet_v2.preprocess_input(img_to_array(pic_pic))
            batch_lst.append(pic)
            i += 1
            # del pic
            # gc.collect()
        labels_data = np.zeros(1)
        batch_data = np.array(batch_lst)
        try:
            yield batch_data, labels_data
        except StopIteration:#https://www.educba.com/python-stopiteration/
            return
#############


#############
def main_func_features():
    #############
    #SETTINGS
    main_directory = drive.mount('/content/gdrive')
    path = '/content/gdrive/MyDrive/task3/'
    #path = 'C:\\Users\\Fabiano\\Desktop\\ETH\\D-MATH MSc\\Elective_curses\\Introduction to Machine Learning\\projects\\3\\'
    #triplet = path +'train_triplets.txt'
    #test = path +'test_triplets.txt'
    pics_dir = path + 'food'
    features_file = path+'features_extracted.pckl'

    #############
    #EXTRACT FEATURES USING PRETRAINED NN, NN INITIALIZATION
    #feature extraction using deep CNN
    pretrained_NN_features = tf.keras.applications.InceptionResNetV2(pooling='avg',include_top=False)
    #https://keras.io/api/applications/inceptionresnetv2/
    pretrained_NN_features.trainable = False

    ftrs_vct_out = Input(shape=input_shape)
    ftrs_vct_in  = Input(shape=input_shape)
    ftrs_vct_out = pretrained_NN_features(ftrs_vct_out)
    pretrained_NN = Model(inputs=ftrs_vct_in, outputs=ftrs_vct_out)

    #############
    # LOAD IMAGES AND RESHAPE THEM
    #zip only for google collab, else comment out
    zip_name = path+'food.zip'
    zip_obj = ZipFile(zip_name, 'r')
    zip_obj.extractall(path)
    zip_obj.close()

    #construct dictionary so that we can easily iterate over idx
    #done couse when unzipping, an extra weird DS_store file is extracted
    #messing with indexing order of pictures in the generator
    pic_ids_dict = {}
    pics_lst = sorted(os.listdir(pics_dir))
    #clean dictionary
    for elmt in pics_lst:
        if '.jpg' not in elmt:
          pics_lst.remove(elmt)

    for i in range(len(pics_lst)):
      if '.jpg' in pics_lst[i]:
        pic_ids_dict[i] = pics_lst[i]
      else:
        print("FOUND NOT JPG")

    #to limit RAM usage we need to utilize a generator
    features_iterable = generator_pictures(pics_dir, pic_ids_dict)
    #https://keras.io/api/models/model_training_apis/
    ftrs = pretrained_NN.predict(features_iterable,steps=10000)
    #feature_preproc_data = (next(batch_data), next(labels_data))

    #############
    #OUTPUT FEATUERS
    # Save features in feature file
    with open(features_file, 'wb') as f:
        pickle.dump(ftrs, f)

#############

if __name__ == "__main__":
    main_func_features()