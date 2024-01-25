import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep


def find_features(target_shape, pretrained_nn, name_pretrained, pretrained_model_prep):

    path = '/Users/jonas/Documents/ETHZ/FS2022/Introduction to machine learning/Project/Project 3/task3/food/'

    pretrained_cnn_tuned = tf.keras.models.Model(pretrained_nn.input,pretrained_nn.layers[-2].output)

    image_file_names = os.listdir(path)
    image_names = [fn[:-4] for fn in image_file_names]
    file_features_name = 'image_features_' + name_pretrained + '.csv'

    
    image_dataframe = pd.DataFrame(columns=image_names)

    for fn in image_file_names:
        if fn[-4:]=='.jpg':

            image_string = tf.io.read_file(path+fn)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize_with_pad(image, target_height=target_shape[1], target_width=target_shape[0])
            image = pretrained_cnn_tuned(pretrained_model_prep(image)[None]) 
            image =  image.numpy().flatten() 
            image_dataframe[fn[:-4]] = pd.Series(image)

    image_dataframe.to_csv(file_features_name, index=False)

res_net = ResNet50(weights="imagenet", input_shape=(224,224) + (3,), include_top=True)
mobile_net = MobileNetV2( weights="imagenet", input_shape=(224,224) + (3,), include_top=True)


print('Exctracting features using ResNet50')
find_features( (224,224), res_net, 'resnet', resnet_prep)
print('Exctracting features using MobileNetV2')
find_features( (224,224), mobile_net, 'mobilenet', mobilenet_prep)
