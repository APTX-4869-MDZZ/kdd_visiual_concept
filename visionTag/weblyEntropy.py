from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import decode_predictions as resnet50_decode_predictions

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import os
import sys
import codecs
import math
sys.path.append('../utils/')
from imageFeature import prepare_img, img_feature

import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
set_session(sess)

def get_random_data():
  positive_data = []
  with open('data/new_positive-finally.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      positive_data.append(line.split('\t')[0])
  random_data = codecs.open('data/random_data.txt', 'w', 'utf-8')
  with open('data/new_negative-random_1-0.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name = line[:-1].split('\t')[0]
      if name in positive_data:
        random_data.write('\t'.join(line.split('\t')[:2]) + '\t' + '1\n')
      else:
        random_data.write('\t'.join(line.split('\t')[:2]) + '\t' + '0\n')

def calculate_distance(X):
  center = X.mean(axis=0)
  X_ = X-center
  distances = np.sum(np.square(X_), axis=-1)
  return distances

def predict_image():
  model = InceptionV3(weights='imagenet', include_top=False)
  names = []
  predict_result = codecs.open('data/webly_entropy_experiment_resnet50.txt', 'w', 'utf-8')
  with open('data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
      names.append(name[:-1])
  with open('data/ground_truth_test_data-4.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, _, label = line[:-1].split('\t')
      index = names.index(name)
      image_predicts = []
      for image in os.listdir('image/' + str(index))[0:10]:
        output = img_feature('image/' + str(index) + '/' + image, inlcude_average_layer=True)
        # img = prepare_img('image/' + str(index) + '/' + image, model_name='InceptionV3')
        # output = model.predict(img)
        predict_id = decode_predictions(output)[0][0]
        image_predicts.append(predict_id[1])
      predict_result.write(name + '\t' + label + '\t' + '\t'.join(image_predicts) + '\n')

def image_features_preprocess():
  names = []
  with open('data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
      names.append(name[:-1])
  with open('data/ground_truth_test_data-4.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, _, label = line[:-1].split('\t')
      index = names.index(name)
      image_vectors = []
      for image in os.listdir('image/' + str(index))[0: 16]:
        output = img_feature('image/' + str(index) + '/' + image, model_name='ResNet50')
        image_vectors.append(output)
      image_vectors = np.squeeze(np.array(image_vectors))
      print(image_vectors.shape)
      np.save('temp2/{}.npy'.format(str(index)), image_vectors)

def image_variance():
  names = []
  with open('data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
      names.append(name[:-1])
  std_0 = []
  std_1 = []
  name_0 = []
  name_1 = []
  with open('data/ground_truth_test_data-4.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, _, label = line[:-1].split('\t')
      index = names.index(name)
      image_vectors = np.load('temp2/{}.npy'.format(str(index)))
      distances = calculate_distance(image_vectors)
      top_distances = np.sort(distances)
      distance_std = np.std(top_distances[0:16], ddof=1)
      if label == '0':
        std_0.append(distance_std)
        name_0.append(name)
      else:
        std_1.append(distance_std)
        name_1.append(name)
  with open('data/std_0.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([str(x) for x in std_0]) + '\n')
  with open('data/std_1.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([str(x) for x in std_1]) + '\n') 
  with open('data/std_0_name.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(name_0) + '\n')
  with open('data/std_1_name.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(name_1) + '\n') 

def analyze_predict_result():
  ent_0 = []
  ent_1 = []
  with open('data/webly_entropy_experiment_resnet50.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name = line[:-1].split('\t')[0]
      label = int(line[:-1].split('\t')[1])
      ids = set()
      predict_result = line[:-1].split('\t')[2:10]
      ent = 0
      for _id in predict_result:
        if not _id in ids:
          ids.add(_id)
          p = predict_result.count(_id) / len(predict_result)
          logp = math.log(p)
          ent -= p * logp
      if label == 0:
        ent_0.append(ent)
      else:
        ent_1.append(ent)
  with open('data/ent_0.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([str(x) for x in ent_0]) + '\n')
  with open('data/ent_1.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([str(x) for x in ent_1]) + '\n')

def plot_ent():
  with open('data/ent_0.txt', 'r', encoding='utf-8') as f:
    ent_0 = [float(x[:-1]) for x in f.readlines()]
    ent_0 = np.array(ent_0)
  with open('data/ent_1.txt', 'r', encoding='utf-8') as f:
    ent_1 = [float(x[:-1]) for x in f.readlines()]
    ent_1 = np.array(ent_1)
  
  plt.figure()
  plt.hist(ent_0, bins=12, color='red', alpha=.4)
  plt.hist(ent_1, bins=12, color='blue', alpha=.4)
  plt.show()

# image_features_preprocess()
image_variance()
# analyze_predict_result()
# model = InceptionV3(weights='imagenet', include_top=False)
# model.summary()