from nltk.corpus import wordnet as wn
import random
import requests
from socket import error as SocketError
import errno
import time
import codecs
import shutil
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_name', default='visionTag/log/VGG16/test-1-0')
parser.add_argument('--action', default='unlabel_data')
parser.add_argument('--experiment', default='InceptionV3')
parser.add_argument('--k', default='')
parser.add_argument('--iteration', default='')
args = parser.parse_args()

def find_best_csv(ks, log_name):
  max_acc = 0
  min_loss = 100
  max_i = 0
  max_k = ''
  max_f1 = 0
  for k in ks.split(','):
    log_name_ = log_name + '-' + k + '.'
    max_acc_ = 0
    min_loss_ = 100
    max_i_ = 0
    max_f1_ = 0
    for i in range(1, 4):
      with open(log_name_ + str(i) + '.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
          if line[0:5] == 'epoch':
            continue
          loss, val_acc, val_f1 = line[:-1].split(',')[2: 5]
          val_acc = float(val_acc)
          loss = float(loss)
          if val_acc > max_acc_ or max_acc_ - val_acc < 1e-4 and loss < min_loss_:
            max_acc_ = val_acc
            min_loss_ = loss
            max_i_ = i
            max_f1_ = val_f1
    if max_acc_ > max_acc or max_acc - max_acc_ < 1e-4 and min_loss_ < min_loss:
      max_acc = max_acc_
      min_loss = min_loss_
      max_i = max_i_
      max_k = k
      max_f1 = max_f1_
  print(max_i, max_acc, max_f1, min_loss, max_k)

imagenet_synset = set()
def get_hyponyms(wn_synset):
  for synset in wn_synset.hyponyms():
    imagenet_synset.add(synset)
    get_hyponyms(synset)

def get_imagenet():
  with open('visionTag/imagenet_wnid.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      synset = wn.synset_from_pos_and_offset('n', int(line[1:-1]))
      imagenet_synset.add(synset)
      # get_hyponyms(synset)
  with open('visionTag/imagenet.txt', 'w', encoding='utf-8') as f:
    for synset in imagenet_synset:
      f.write(synset.name() + '\n')

def prepare_data():
  data = []
  with open('visionTag/data/ILSVRC_data.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      data.append(line[:-1] + '\t1\n')
  positive_samples_num = len(data)

  wordnets = []
  with open('visionTag/data/unlabel_data.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      wordnets.append(line[:-1])

  for line in random.sample(wordnets, positive_samples_num):
    data.append(line[:-1] + '\t0\n')

  random.shuffle(data)
  with open('visionTag/data/ILSVRC_train.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(data))

def check_data():
  names = []
  definitions = []
  labels = []
  with open('visionTag/data.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, definition, label = line[:-1].split('\t')
      labels.append(label)
      names.append(name)
      definitions.append(definition)

  negetive_lemmas = 0
  positive_lemmas = 0
  for name, label in zip(names, labels):
    if len(wn.synset(name).member_holonyms()) == 0 :
      if int(label) == 0:
        negetive_lemmas += 1
      else:
        positive_lemmas += 1
  print(negetive_lemmas, positive_lemmas)

  positive_min_depth = 99999
  negetive_min_depth = 99999
  positive_max_depth = 0
  negetive_max_depth = 0
  for name, label in zip(names, labels):
    if int(label) == 0:
      if wn.synset(name).min_depth() < negetive_min_depth:
        negetive_min_depth = wn.synset(name).min_depth()
      if wn.synset(name).min_depth() > negetive_max_depth:
        negetive_max_depth = wn.synset(name).min_depth()
    else:
      if wn.synset(name).min_depth() < positive_min_depth:
        positive_min_depth = wn.synset(name).min_depth()
      if wn.synset(name).min_depth() > positive_max_depth:
        positive_max_depth = wn.synset(name).min_depth()
  print(positive_min_depth, positive_max_depth, negetive_min_depth, negetive_max_depth)

# get_imagenet()
# prepare_data()

def split_test_train():
  positive_data = []
  negetive_data = []
  with open('visionTag/data/ILSVRC_data-2.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, definition, label = line[:-1].split('\t')
      if int(label) == 1:
        positive_data.append((name, definition))
      else:
        negetive_data.append((name, definition))
  print(len(positive_data))
  print(len(negetive_data))

  test_data = []
  train_data = []
  total_num = len(positive_data)
  test_num = total_num // 5
  j = 1
  indexs = [x for x in range(total_num)]
  random.shuffle(indexs)
  for i in indexs:
    if j <= test_num:
      test_data.append(positive_data[i][0] + '\t' + positive_data[i][1] + '\t1\n')
      test_data.append(negetive_data[i][0] + '\t' + negetive_data[i][1] + '\t0\n')
    else:
      train_data.append(positive_data[i][0] + '\t' + positive_data[i][1] + '\t1\n')
      train_data.append(negetive_data[i][0] + '\t' + negetive_data[i][1] + '\t0\n')
    j += 1
  with open('visionTag/data/ILSVRC_train-2.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(train_data))
  with open('visionTag/data/ILSVRC_test-2.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(test_data))

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
def safe_request(url):
  for _ in range(3):
    try:
      item_response = requests.get(url, headers=headers)
      break
    except Exception as e:
      print(url)
      print(e)
  return item_response

def filter_imagenet_synset():
  imagenet_filtered = codecs.open('visionTag/imagenet_filtered.txt', 'a', encoding='utf-8')
  with open('visionTag/imagenet_wnid.txt', 'r', encoding='utf-8') as f:
    imagenet_synset = f.readlines()
  for synset in imagenet_synset[13007:]:
    response = safe_request('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='+synset[:-1])
    if len(response.text.strip()) > 0:
      imagenet_filtered.write(synset)
    print(synset)

def ILSVRC_data():
  with open('visionTag/imagenet_words.txt', 'r', encoding='utf-8') as f:
    imagenet_words_file = f.readlines()
  imagenet_words = dict()
  for line in imagenet_words_file:
    wnid, word = line[:-1].split('\t')
    imagenet_words[word] = wnid
  with open('visionTag/ILSVRC_word.txt', 'r', encoding='utf-8') as f:
    ILSVRC_data = f.readlines()
  ILSVRC_wnid = []
  for synset_name in ILSVRC_data:
    ILSVRC_wnid.append(imagenet_words[synset_name[:-1]])
  with open('visionTag/ILSVRC_wnid.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(ILSVRC_wnid) + '\n')

def ground_truth():
  with open('visionTag/data/positive_ground_true.txt', 'r', encoding='utf-8') as f:
    positives = f.readlines()
  with open('visionTag/data/negative_ground_true.txt', 'r', encoding='utf-8') as f:
    negatives = f.readlines()
  test_data = []
  for line in positives:
    test_data.append(line[:-1] + '\t1\n')
  for line in negatives:
    test_data.append(line[:-1] + '\t0\n')
  random.shuffle(test_data)
  with open('visionTag/data/ground_truth_test_data.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(test_data))

def unique_definition():
  definitions = set()
  ILSVRC_data = []
  with open('visionTag/data/ILSVRC_wnid.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      synset = wn.synset_from_pos_and_offset('n', int(line[1:-1]))
      if synset.definition() in definitions:
        continue
      else:
        definitions.add(synset.definition())
        ILSVRC_data.append(synset.name() + '\t' + synset.definition() + '\n')
  with open('visionTag/data/ILSVRC_data.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(ILSVRC_data))

  new_wordnet = []
  with open('visionTag/data/wordnets.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      synset = wn.synset(line[:-1])
      if synset.definition() in definitions:
        continue
      else:
        definitions.add(synset.definition())
        new_wordnet.append(synset.name() + '\t' + synset.definition() + '\n')
  with open('visionTag/data/unlabel.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(new_wordnet))

def top_negative():
  negatives = []
  with open('visionTag/data/new_negative_with_image-0.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, definition, score = line[:-1].split('\t')
      score = float(score[1:-1])
      negatives.append((name, definition, score))
  negatives = sorted(negatives, key=lambda x: x[2])
  with open('visionTag/data/negative_temp.txt', 'w', encoding='utf-8') as f:
    for negative in negatives[:3000]:
      f.write(negative[0] + '\t' + negative[1] + '\t' + str(negative[2]) + '\n')

def top_positive():
  positives = []
  with open('visionTag/data/new_positive-1.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, definition, score = line[:-1].split('\t')
      score = float(score[1:-1])
      positives.append((name, definition, score))
  positives = sorted(positives, key=lambda x: x[2], reverse=True)
  with open('visionTag/data/positive_temp.txt', 'w', encoding='utf-8') as f:
    for positive in positives[:3000]:
      f.write(positive[0] + '\t' + positive[1] + '\t' + str(positive[2]) + '\n')

def reliable_negative():
  reliable_negatives = []
  with open('visionTag/predict_count.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines()[:999]:
      name = line[:-1].split('\t')[0]
      reliable_negatives.append(name + '\t' + wn.synset(name).definition() + '\n')
  with open('visionTag/data/negative_temp.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(reliable_negatives))

def reliable_train_data():
  data = []
  with open('visionTag/data/ILSVRC_data.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      data.append(line[:-1] + '\t1\n')
  with open('visionTag/data/negative_temp.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines()[:999]:
      name, definition = line[:-1].split('\t')[0:2]
      data.append(name + '\t' + definition + '\t0\n')
  random.shuffle(data)
  with open('visionTag/data/ILSVRC_train-only_webly-0.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(data))

def new_reliable_train_data():
  data = []
  with open('visionTag/data/ILSVRC_train-1.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      data.append(line)
  text_negatives = []
  with open('visionTag/data/negative_temp.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      text_negatives.append(line.split('\t')[0])
  text_positives = []
  with open('visionTag/data/positive_temp.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      text_positives.append(line.split('\t')[0])

  names = []
  with open('visionTag/predict_count.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      names.append(line.split('\t')[0])
  i = 0
  for name in names:
    if name in text_negatives:
      data.append(name + '\t' + wn.synset(name).definition() + '\t0\n')
      i += 1
      if i == 1000:
        break
  i = 0
  for name in reversed(names):
    if name in text_positives:
      data.append(name + '\t' + wn.synset(name).definition() + '\t1\n')
      i += 1
      if i == 1000:
        break
  random.shuffle(data)
  with open('visionTag/data/ILSVRC_train-2.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(data))

def predict_rank():
  text_rank = {}
  i = 0
  with open('visionTag/data/positive_temp.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      text_rank[line.split('\t')[0]] = i
      i += 1
  names = []
  with open('visionTag/data/negative_temp.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      names.append(line.split('\t')[0])
  for name in reversed(names):
    text_rank[line.split('\t')[0]] = i
    i += 1

  i = 0
  names = []
  with open('visionTag/predict_count.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      names.append(line.split('\t')[0])
  for name in reversed(names):
    if name in text_rank:
      text_rank[name] = (text_rank[name] + i) / 2
      i += 1

  ranks = sorted(text_rank.items(), key=lambda x: x[1], reverse=True)
  with open('visionTag/predict_rank.txt', 'w', encoding='utf-8') as f:
    for rank in ranks:
      f.write(rank[0] + '\t' + str(rank[1]) + '\n')

def preprocess_image_data():
  # from keras.applications.vgg16 import VGG16
  from keras.applications.resnet50 import ResNet50
  from keras.layers import Dense, GlobalAveragePooling2D
  from keras.models import Model
  import numpy as np
  import sys
  import os
  sys.path.append('utils/')
  from imageFeature import prepare_img

  from keras.backend.tensorflow_backend import set_session
  import tensorflow as tf

  conf = tf.ConfigProto()
  conf.gpu_options.allow_growth = True
  sess = tf.Session(config=conf)
  set_session(sess)

  base_model = ResNet50(weights='imagenet', include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  model = Model(inputs=base_model.input, outputs=x)
  i = 0
  for folder_name in os.listdir('visionTag/image'):
    if os.path.exists('visionTag/ResNet50/' + folder_name + '.npy') or folder_name == 'feature_iteration0':
      continue
    image_feature = []
    for image in os.listdir('visionTag/image/' + folder_name)[:16]:
      image_feature.append(model.predict(prepare_img('visionTag/image/' + folder_name + '/' + image, model_name='ResNet50')))
    image_feature = np.squeeze(np.array(image_feature))
    with open('visionTag/ResNet50/' + folder_name + '.npy', 'wb') as f:
      np.save(f, image_feature)
    i += 1
    if i % 100 == 0:
      print(i, 'concept preprocessed.')

def unlabeled_data(k, iteration):
  experiment = args.experiment
  print('unlabeled_data')
  print('visionTag/data/{}/ILSVRC_train-{}.txt'.format(experiment, k))
  print('visionTag/data/{}/unlabel_data-{}.txt'.format(experiment, int(iteration)-1))
  print('visionTag/data/{}/unlabel_data-{}.txt'.format(experiment, iteration))
  ILSVRC_data = []
  with open('visionTag/data/{}/ILSVRC_train-{}.txt'.format(experiment, k), 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, _, _ = line.split('\t')
      ILSVRC_data.append(name)
  data = []
  with open('visionTag/data/{}/unlabel_data-{}.txt'.format(experiment, int(iteration)-1), 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name = line.split('\t')[0]
      if not name in ILSVRC_data:
        synset = wn.synset(name)
        data.append(synset.name() + '\t' + synset.definition() + '\t0\n')
  with open('visionTag/data/{}/unlabel_data-{}.txt'.format(experiment, iteration), 'w', encoding='utf-8') as f:
    f.write(''.join(data))

def new_train_data(_k, iteration):
  experiment = args.experiment
  print('new_train_data')
  print('visionTag/data/{}/ILSVRC_train-{}.txt'.format(experiment, _k))
  print('visionTag/data/{}/new_positive-{}.txt'.format(experiment, iteration))
  print('visionTag/data/{}/new_negative-{}.txt'.format(experiment, iteration))
  count_positive = 0
  ILSVRC_data = []
  with open('visionTag/data/{}/ILSVRC_train-{}.txt'.format(experiment, _k), 'r', encoding='utf-8') as f:
    for line in f.readlines():
      if line[-2] == '1':
        count_positive += 1
      ILSVRC_data.append(line)
  
  positives = []
  with open('visionTag/data/{}/new_positive-{}.txt'.format(experiment, iteration), 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, definition, score = line[:-1].split('\t')
      score = float(score[1:-1])
      positives.append((name + '\t' + definition + '\t1\n', score))
  positives = sorted(positives, key=lambda x:x[1], reverse=True)

  negatives = []
  with open('visionTag/data/{}/new_negative-{}.txt'.format(experiment, iteration), 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, definition, score = line[:-1].split('\t')
      score = float(score[1:-1])
      negatives.append((name + '\t' + definition + '\t0\n', score))
  negatives = sorted(negatives, key=lambda d:d[1])

  for k in [1.5, 2, 2.5, 3, 4]:
    ILSVRC_data = []
    with open('visionTag/data/{}/ILSVRC_train-{}.txt'.format(experiment, _k), 'r', encoding='utf-8') as f:
      ILSVRC_data = f.readlines()
    new_count = k*count_positive - count_positive
    if new_count > len(negatives) or new_count > len(positives):
      break
    for i in range(int(k*count_positive - count_positive)):
      ILSVRC_data.append(positives[i][0])
      ILSVRC_data.append(negatives[i][0])
    random.shuffle(ILSVRC_data)
    with open('visionTag/data/{}/ILSVRC_train-k_{}-{}.txt'.format(experiment, k, int(iteration)+1), 'w', encoding='utf-8') as f:
      f.write(''.join(ILSVRC_data))


if __name__=="__main__":
  if args.action == 'find_best_csv':
    find_best_csv(args.k, args.log_name)
  if args.action == 'unlabeled_data':
    unlabeled_data(args.k, args.iteration)
  if args.action == 'new_train_data':
    new_train_data(args.k, args.iteration)
  if args.action == 'preprocess_image_data':
    preprocess_image_data()