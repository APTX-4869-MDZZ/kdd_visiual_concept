from keras.models import Model
from keras.layers import Dense, Input, Embedding, LSTM, Lambda, Activation, Dropout, Concatenate, Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import optimizers
from gensim.models import Word2Vec

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
set_session(sess)

import numpy as np

import os
import codecs
import random
import sys

import bert
bert.set_language('en')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', default='visionTag/data/ILSVRC_train.txt')
parser.add_argument('--test', default='visionTag/data/ground_truth_test_data-1.txt')
parser.add_argument('--with_image', action="store_true")
parser.add_argument('--unlabel', default='visionTag/data/unlabel_data.txt')
parser.add_argument('--new_positive', default='visionTag/data/new_positive.txt')
parser.add_argument('--new_negative', default='visionTag/data/new_negative.txt')
parser.add_argument('--model', default='visionTag/models/only_definition_ILSVRC_washed-bert-0.h5')
parser.add_argument('--log', default='visionTag/log/log.txt')
parser.add_argument('--action', default='train_with_bert')
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('--errors', default='visionTag/data/ground_truth_errors.txt')
parser.add_argument('--experiment', default='InceptionV3')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--load', default=None)
args = parser.parse_args()

'''
for train:
python visionTag/train.py --action train_with_bert --train visionTag/data/ILSVRC_train-1.txt --test visionTag/data/ground_truth_test_data-1.txt --model visionTag/models/only_definition_ILSVRC_washed-bert-1.2.h5 --log visionTag/log/log_only_definition_trian-1_test-1_model-1.2.csv

for train_with_image:
python visionTag/train.py --action train_with_bert --with_image --train visionTag/data/ILSVRC_train-1.txt --test visionTag/data/ground_truth_test_data-1.txt --model visionTag/models/definition_image_ILSVRC_washed-bert-2.3.h5 --log visionTag/log/log_definition_image_trian-1_test-1_model-1.1.csv

for expectation:
python visionTag/train.py --action expectation --unlabel visionTag/data/unlabel_data-1.txt --new_positive visionTag/data/new_positive-1.txt --new_negative visionTag/data/new_negative-1.txt --model visionTag/models/only_definition_ILSVRC_washed-bert-1.h5

for eval:
python visionTag/train.py --action eval --test visionTag/data/ground_truth_test_data-1.txt --model visionTag/models/only_definition_ILSVRC_washed-bert-2.1.h5
for eval_with_image:
python visionTag/train.py --action eval --with_image --test visionTag/data/ground_truth_test_data-1.txt --model visionTag/models/definition_image_ILSVRC_washed-bert-2.h5
'''

from sklearn.metrics import f1_score, recall_score, precision_score
from keras.callbacks import Callback

class Metrics(Callback):
  def __init__(self, valid_data):
    super(Metrics, self).__init__()
    self.valid_data = []
    self.valid_data.append(valid_data[0])
    self.valid_data.append(np.array(valid_data[1]))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    val_predict = self.model.predict(self.valid_data[0])
    val_predict[val_predict > 0.5] = 1
    val_predict[val_predict <= 0.5] = 0
    val_targ = self.valid_data[1]
    if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
        val_targ = np.argmax(val_targ, -1)

    _val_f1 = f1_score(val_targ, val_predict, average='macro')
    _val_recall = recall_score(val_targ, val_predict, average='macro')
    _val_precision = precision_score(val_targ, val_predict, average='macro')

    logs['val_f1'] = _val_f1
    logs['val_recall'] = _val_recall
    logs['val_precision'] = _val_precision
    print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
    return

def prepare_bert_data(data_file):
  names = []
  texts = []
  labels = []

  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name, sentence, label = line[:-1].split('\t')
      names.append(name)
      texts.append(sentence)
      labels.append(int(label))

  return names, texts, labels

def prepare_image_data(data_file, with_batch=False, i=0):
  names = []
  images = []
  vector_shape = 2048
  if args.experiment == 'VGG16':
    vector_shape = 512

  with open('visionTag/data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
      names.append(name[:-1])

  data_indexs = []
  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      name = line.split('\t')[0]
      index = names.index(name)
      data_indexs.append(index)
  if with_batch:
    batch = 10000
  else:
    batch = len(data_indexs)
  for index in data_indexs[i*batch: min(i*batch + batch, len(data_indexs))]:
    with open('visionTag/' + args.experiment + '/' + str(index) + '.npy', 'rb') as f:
      image_feature = np.load(f)
    # try:
      images.append(image_feature.reshape(16*vector_shape, ))
    # except:
    #   print(index)
      
  images = np.squeeze(np.array(images))

  return images

def get_model(with_image=False):
  max_len = 64
  bert_inputs = bert.get_bert_inputs(max_len)
  bert_output = bert.BERTLayer(n_fine_tune_vars=3)(bert_inputs)
  x = Dense(256, activation='relu')(bert_output)

  if with_image:
    vector_shape = 2048
    if args.experiment == 'VGG16':
      vector_shape = 512
    image_input = Input(shape=(16*vector_shape,))
    image = Dense(256, activation='relu')(image_input)
    x = Concatenate()([x, image])

    bert_inputs.append(image_input)
  x = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=bert_inputs, outputs=x)
  return model

def generate_batch(train_inputs, data_file):
  names = []

  with open('visionTag/data/washed_wordnet.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
      names.append(name[:-1])

  while True:
    i = 0
    with open(data_file, 'r', encoding='utf-8') as f:
      for line in f.readlines():
        name, _, label = line[:-1].split('\t')
        label = int(label)
        index = names.index(name)
        with open('visionTag/' + args.experiment + '/' + str(index) + '.npy', 'rb') as f:
          image_feature = np.load(f)
          vector_shape = 2048
          if args.experiment == 'VGG16':
            vector_shape = 512
          image_feature = image_feature.reshape(16*vector_shape, )
          print(train_inputs[0][i].shape)
          x = [train_inputs[0][i], train_inputs[1][i], train_inputs[2][i], image_feature]
          yield (x, label)
        i = i + 1

def train_with_bert(with_image=False):
  checkpoint_file = args.model
  _, train_text, train_label = prepare_bert_data(args.train)
  _, test_text, test_label = prepare_bert_data(args.test)
  max_len = 64
  train_inputs, test_inputs = map(lambda x:bert.convert_sentences(x, max_len), [train_text, test_text])
  model = get_model(with_image)
  
  if args.load != 'NULL':
    model.load_weights(args.load)

  epochs = args.epoch
  batch_size = 32
  total_steps = epochs*train_inputs[0].shape[0]//batch_size
  steps_per_epoch = train_inputs[0].shape[0]//batch_size
  lr_scheduler, opt_lr = bert.get_suggested_scheduler(init_lr=args.lr, total_steps=total_steps)
  optimizer = bert.get_suggested_optimizer(opt_lr)
  
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  model.summary()

  if with_image:
    train_image = prepare_image_data(args.train)
    test_image = prepare_image_data(args.test)
    train_inputs.append(train_image)
    test_inputs.append(test_image)
  checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  # ck_callback = ModelCheckpoint(checkpoint_file, monitor='val_acc', mode='max', verbose=2, save_best_only=True, save_weights_only=True)
  # tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch=0)
  csv_logger = CSVLogger(args.log)
  # model.fit_generator(generate_batch(train_inputs, args.train), epochs=epochs, steps_per_epoch=steps_per_epoch,
  #     validation_data=(test_inputs, test_label), callbacks=[lr_scheduler, checkpoint, csv_logger])
  metrics = Metrics(valid_data=(test_inputs, test_label))
  model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size, 
		  validation_data=(test_inputs, test_label), callbacks=[lr_scheduler, metrics, checkpoint, csv_logger])

def analyze_error():
  model = get_model(args.with_image)
  model.load_weights(args.model)

  test_name, test_text, test_label = prepare_bert_data(args.test)

  errors = []
  max_len = 64
  test_inputs = bert.convert_sentences(test_text, max_len)
  ys = model.predict(test_inputs)
  for i in range(len(test_label)):
    if np.around(ys[i]) != test_label[i]:
      errors.append(test_name[i] + '\t' + test_text[i] + '\t' + str(test_label[i]) + '\n')
  with open(args.errors, 'w', encoding='utf-8') as f:
    f.write(''.join(errors))

def predict_unlabel_data(with_image=False):
  test_name, test_text, _ = prepare_bert_data(args.unlabel)
  model = get_model(with_image)
  model.load_weights(args.load)
  max_len = 64

  new_positive = []
  new_negative = []
  test_inputs = bert.convert_sentences(test_text, max_len)

  batch = 10000
  ys = []
  for i in range(len(test_text) // batch + 1):
    sub_test_inputs = [test_inputs[0][i*batch: min(i*batch + batch, len(test_text))],
      test_inputs[1][i*batch: min(i*batch + batch, len(test_text))], 
      test_inputs[2][i*batch: min(i*batch + batch, len(test_text))]]
    if with_image:
      sub_test_inputs.append(prepare_image_data(args.unlabel, with_batch=True, i=i))
    sub_ys = model.predict(sub_test_inputs)
    ys.extend(sub_ys)
  for i in range(len(test_text)):
    y = ys[i]
    if np.around(y) == 1:
      new_positive.append(test_name[i] + '\t' + test_text[i] + '\t' + str(y) + '\n')
    else:
      new_negative.append(test_name[i] + '\t' + test_text[i] + '\t' + str(y) + '\n')
  with open(args.new_positive, 'w', encoding='utf-8') as f:
    f.write(''.join(new_positive))
  with open(args.new_negative, 'w', encoding='utf-8') as f:
    f.write(''.join(new_negative))


def eval_model(with_image=False):
  from sklearn.metrics import classification_report, accuracy_score
  model = get_model(with_image)
  model.load_weights(args.model)

  _, test_text, test_label = prepare_bert_data(args.test)
  
  max_len = 64
  test_inputs = bert.convert_sentences(test_text, max_len)
  
  if with_image:
    test_image = prepare_image_data(args.test)
    test_inputs.append(test_image)

  ys = model.predict(test_inputs)
  y_pred = []
  for y in ys:
    y_pred.append(np.around(y).item())
  print(classification_report(test_label, y_pred))
  print(accuracy_score(test_label, y_pred))

def test(with_image=False):
  model = get_model(with_image)
  model.load_weights(args.model)

  test_name, test_text, _ = prepare_bert_data(args.test)
  max_len = 64
  test_inputs = bert.convert_sentences(test_text, max_len)
  if with_image:
    test_image = prepare_image_data(args.test)
    test_inputs.append(test_image)
  ys = model.predict(test_inputs)
  i = 0
  with open('visionTag/predict_result.txt', 'w', encoding='utf-8') as f:
    for y in ys:
      y_pred = np.around(y).item()
      f.write(test_name[i] + '\t' + test_text[i] + '\t' + str(y_pred) + '\n')
      i = i + 1

if __name__ == "__main__":
  print(sys.argv)
  if not os.path.exists('./visionTag/log'):
    os.makedirs('./visionTag/log')
  if not os.path.exists('./visionTag/model'):
    os.makedirs('./visionTag/model')
    
  if args.with_image:
    print('with image')
  if args.action == 'train_with_bert':
    train_with_bert(args.with_image)
  if args.action == 'predict':
    predict_unlabel_data(args.with_image)
  if args.action == 'eval':
    eval_model(args.with_image)
  if args.action == 'analysis':
    analyze_error()
  if args.action == 'test':
    test(args.with_image)