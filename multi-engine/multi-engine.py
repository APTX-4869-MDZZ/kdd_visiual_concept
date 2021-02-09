from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.models import Model

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os
import re
import shutil

from absl import flags
from absl import app
from absl import logging

from imageEngine import ImageSearchEngine

flags.DEFINE_string('data_path', 'data/image/engine/', '')
flags.DEFINE_string('mode', 'download', '')
flags.DEFINE_integer('k_engine', 20, '')
flags.DEFINE_integer('topk', 5, '')
FLAGS = flags.FLAGS

def prepare_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return vgg_preprocess(img_array_expanded_dims)

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    feature = Dense(1024, activation='relu', name='img_feature')(x)
    model = Model(inputs=base_model.input, outputs=feature)
    
    for layer in model.layers:
        layer.trainable=False
    
    return model

def move_image(files):
    for file_name in files:
        path_list = file_name.split('/')
        path_list[-4] = 'filter'
        path_list[-1] = path_list[-3]+path_list[-1]
        del path_list[-3]
        if not os.path.exists('/'.join(path_list[0: -1])):
            os.mkdir('/'.join(path_list[0: -1]))
        shutil.copy(file_name, '/'.join(path_list))

def get_topk_image(entity):
    images = []
    files = []
    for engine in os.listdir(FLAGS.data_path):
        for image in os.listdir(FLAGS.data_path + engine + '/' + entity):
            try:
                image_path = FLAGS.data_path + engine + '/' + entity + '/' + image
                images.append(prepare_img(image_path))
                files.append(image_path)
            except:
                print(engine, image)
    images = np.squeeze(np.array(images))
    files = np.squeeze(np.array(files))
    
    model = create_model()
    features = model.predict(images)
    similarity = cosine_similarity(features)
    similarity = np.sum((similarity > 0.9)*similarity, axis=0)
    args = (-similarity).argsort()[: FLAGS.topk]

    print(files[args])
    move_image(files[args])

def main(_):
    imageEngine = ImageSearchEngine(FLAGS.k_engine, ['baidu', 'google', 'bing', 'yahoo', 'aol', 'duckduckgo'], 'data/image/engine/')
    # imageEngine = ImageSearchEngine(FLAGS.k_engine, ['bing'], 'data/image/engine/')

    popular_entity = None
    with open('popular_entity.txt', 'r', encoding='utf-8') as file:
        popular_entity = file.readlines()
    
    if FLAGS.mode == 'download':
        for index, entity in enumerate(popular_entity[12: 13]):
            colon_positin = entity.find(':')
            entity_name = entity[0: colon_positin]
            class_ = entity[colon_positin+1:]
            imageEngine.get_topk_from_engine(entity_name, class_)
            print(index, entity_name)
    elif FLAGS.mode == 'filter':
        for entity in popular_entity[12: 1000]:
            colon_positin = entity.find(':')
            entity_name = entity[0: colon_positin]
            entity_name = re.sub(r'[\"|\/|\?|\*|\:|\||\\|\<|\>]', ' ', entity_name)
            get_topk_image(entity_name)

if __name__ == "__main__":
    app.run(main)