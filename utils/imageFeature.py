import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.models import Model

target_sizes = {
  'InceptionV3': (299, 299),
  'ResNet50': (224, 224),
  'VGG16': (224, 224)
}
model = None
_model = None
def prepare_img(img_path, model_name='InceptionV3'):
  img = image.load_img(img_path, target_size=target_sizes[model_name])
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  if model_name == 'InceptionV3':
    return inception_v3_preprocess(x)
  if model_name == 'ResNet50':
    return resnet50_preprocess(x)
  if model_name == 'VGG16':
    return vgg16_preprocess(x)

def img_feature(img_path, model_name='InceptionV3', inlcude_average_layer=False):
  global model
  global _model
  if not model:
    if model_name == 'InceptionV3':
      model = InceptionV3(weights='imagenet', include_top=False)
    if model_name == 'VGG16':
      model = VGG16(weights='imagenet', include_top=False)
    if model_name == 'ResNet50':
      model = ResNet50(weights='imagenet', include_top=False)
  if inlcude_average_layer:
    if not _model:
      _model = model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    return _model.predict(prepare_img(img_path, model_name))

  return model.predict(prepare_img(img_path, model_name)).reshape((7*7*2048, 1))