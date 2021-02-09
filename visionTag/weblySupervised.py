import re
import sys
import os
import time
import math
import codecs
import numpy as np
import tqdm
from nltk.corpus import wordnet as wn

sys.path.append('multi-engine/')
from imageEngine import ImageSearchEngine
sys.path.append('utils/')
from imageFeature import prepare_img, img_feature

from keras.applications.inception_v3 import InceptionV3, decode_predictions

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
set_session(sess)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=int)
parser.add_argument('-j', type=int)
parser.add_argument('--image_path', type=str)
parser.add_argument('--candidate', type=str)
parser.add_argument('--action', type=str)
args = parser.parse_args()

def download_image():
  print('start downloading images.')
  with open('visionTag/data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    names = f.read().split('\n')
  imageEngine = ImageSearchEngine(20, ['google'], args.image_path)
  for index, old_name in enumerate(names[args.i: args.j]):
    if os.path.exists(args.image_path + '/' + str(index + args.i)):
      continue
    definition = wn.synset(old_name).definition()
    name = re.match(r'(.*)\.n', old_name).group(1)
    name = name.replace('_', ' ')
    imageEngine.get_topk_from_engine(name, definition, folder_name=str(index + args.i))
    time.sleep(3)

def check_image_score():
  image_scores = []
  with open('visionTag/image_scores.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, score = line[:-1].split('\t')
      score = float(score)
      image_scores.append((name, score))
  image_scores = sorted(image_scores, key=lambda x: x[1], reverse=True)
  with open('visionTag/image_scores.txt', 'w', encoding='utf-8') as f:
    for image_score in image_scores:
      f.write(image_score[0] + '\t' + str(image_score[1]) + '\n')

def predict_image(file_name, image_path):
  with open('visionTag/data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    names = f.read().split('\n')
  predict_result = codecs.open('visionTag/data/image_predict_result.txt', 'w', 'utf-8')
  model = InceptionV3(weights='imagenet')
  with open(file_name, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name = line.split('\t')[0]
      index = names.index(name)
      image_predicts = []
      for image in os.listdir(image_path + str(index)):
        img = prepare_img(image_path + str(index) + '/' + image, model_name='InceptionV3')
        predict_id = decode_predictions(model.predict(img))[0][0]
        image_predicts.append(predict_id[0])
      predict_result.write(name + '\t' + '\t'.join(image_predicts) + '\n')
    
def analyze_predict_result():
  count_class = []
  with open('visionTag/predict_result.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name = line[:-1].split('\t')[0]
      ids = set()
      predict_result = line[:-1].split('\t')[1:]
      ent = 0
      for _id in predict_result:
        if not _id in ids:
          ids.add(_id)
          p = predict_result.count(_id) / len(predict_result)
          logp = math.log(p)
          ent -= p * logp
      count_class.append((name, ent))
  count_class = sorted(count_class, key=lambda x:x[1], reverse=True)
  with open('visionTag/predict_count.txt', 'w', encoding='utf-8') as f:
    for predict_ent in count_class:
      f.write(predict_ent[0] + '\t' + str(predict_ent[1]) + '\n')

if __name__ == '__main__':
  if args.action == 'download':
    download_image()
  if args.action == 'filter':
    predict_image(args.candidate, args.image_path)