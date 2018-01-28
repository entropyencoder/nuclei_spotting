# coding: utf-8

import tensorflow as tf
import keras

# Disallow eager use of GPU memory
if 'tensorflow' == keras.backend.backend():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.visible_device_list = "0"
	keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# Prepare training & test datasets
import cv2
import os
import numpy as np

def make_df(train_path, test_path, img_size):
  train_ids = next(os.walk(train_path))[1]
  test_ids = next(os.walk(test_path))[1]
  X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
  Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
  for i, id_ in enumerate(train_ids):
    path = train_path + id_
    img = cv2.imread(path + '/images/' + id_ + '.png')
    img = cv2.resize(img, (img_size, img_size))
    X_train[i] = img
    mask = np.zeros((img_size, img_size, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
      mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
      mask_ = cv2.resize(mask_, (img_size, img_size))
      mask_ = mask_[:, :, np.newaxis]
      mask = np.maximum(mask, mask_)
    Y_train[i] = mask
  X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
  sizes_test = []
  for i, id_ in enumerate(test_ids):
    path = test_path + id_
    img = cv2.imread(path + '/images/' + id_ + '.png')
    sizes_test.append([img.shape[0], img.shape[1]])
    img = cv2.resize(img, (img_size, img_size))
    X_test[i] = img

  return X_train, Y_train, X_test, sizes_test


# Define generator. Using keras ImageDataGenerator.
# You can change the method of data augmentation by changing data_gen_args.
from keras.preprocessing.image import ImageDataGenerator

def generator(xtr, xval, ytr, yval, batch_size):
  data_gen_args = dict(horizontal_flip=True,
                       vertical_flip=True,
                       rotation_range=90.,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       zoom_range=0.1)
  image_datagen = ImageDataGenerator(**data_gen_args)
  mask_datagen = ImageDataGenerator(**data_gen_args)
  image_datagen.fit(xtr, seed=7)
  mask_datagen.fit(ytr, seed=7)
  image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
  mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
  train_generator = zip(image_generator, mask_generator)

  val_gen_args = dict()
  image_datagen_val = ImageDataGenerator(**val_gen_args)
  mask_datagen_val = ImageDataGenerator(**val_gen_args)
  image_datagen_val.fit(xval, seed=7)
  mask_datagen_val.fit(yval, seed=7)
  image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
  mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
  val_generator = zip(image_generator_val, mask_generator_val)

  return train_generator, val_generator


# Define metric functions
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy

def iou_metric(y_true_in, y_pred_in, print_table=False):
  labels = label(y_true_in > 0.5)
  y_pred = label(y_pred_in > 0.5)

  true_objects = len(np.unique(labels))
  pred_objects = len(np.unique(y_pred))

  intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

  # Compute areas (needed for finding the union between all objects)
  area_true = np.histogram(labels, bins=true_objects)[0]
  area_pred = np.histogram(y_pred, bins=pred_objects)[0]
  area_true = np.expand_dims(area_true, -1)
  area_pred = np.expand_dims(area_pred, 0)

  # Compute union
  union = area_true + area_pred - intersection

  # Exclude background from the analysis
  intersection = intersection[1:, 1:]
  union = union[1:, 1:]
  union[union == 0] = 1e-9

  # Compute the intersection over union
  iou = intersection / union

  # Precision helper function
  def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

  # Loop over IoU thresholds
  prec = []
  if print_table:
    print("Thresh\tTP\tFP\tFN\tPrec.")
  for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    if (tp + fp + fn) > 0:
      p = tp / (tp + fp + fn)
    else:
      p = 0
    if print_table:
      print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)

  if print_table:
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
  return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
  batch_size = y_true_in.shape[0]
  metric = []
  for batch in range(batch_size):
    value = iou_metric(y_true_in[batch], y_pred_in[batch])
    metric.append(value)
  return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
  metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
  return metric_value


# Define a loss function
def dice_coef(y_true, y_pred):
  smooth = 1.
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
  return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# Define a run-length encoding function
from skimage.morphology import label

def rle_encoding(x):
  dots = np.where(x.T.flatten() == 1)[0]
  run_lengths = []
  prev = -2
  for b in dots:
    if (b > prev + 1): run_lengths.extend((b + 1, 0))
    run_lengths[-1] += 1
    prev = b
  return run_lengths

def prob_to_rles(x, cutoff=0.5):
  lab_img = label(x > cutoff)
  for i in range(1, lab_img.max() + 1):
    yield rle_encoding(lab_img == i)


# A main procedure to train & predict
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from models import *

if __name__ == "__main__":
  img_size = 256
  batch_size = 32
  train_path = '../input/stage1_train/'
  test_path = '../input/stage1_test/'

  X_train, Y_train, X_test, sizes_test = make_df(train_path, test_path, img_size)
  xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
  train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size)

  model = Unet(img_size)
  model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[my_iou_metric])

  earlystopper = EarlyStopping(patience=10, verbose=1)
  checkpointer = ModelCheckpoint('model_highest_val.h5', verbose=1, save_best_only=True)
  model.fit_generator(train_generator, steps_per_epoch=len(xtr) / 6, epochs=100,
                      validation_data=val_generator, validation_steps=len(xval) / batch_size,
                      callbacks=[earlystopper, checkpointer])

  model = load_model('model_highest_val.h5',
                     custom_objects={'my_iou_metric': my_iou_metric, 'bce_dice_loss': bce_dice_loss})
  preds_test = model.predict(X_test, verbose=1)

  preds_test_upsampled = []
  for i in range(len(preds_test)):
    preds_test_upsampled.append(cv2.resize(preds_test[i],
                                           (sizes_test[i][1], sizes_test[i][0])))

  test_ids = next(os.walk(test_path))[1]
  new_test_ids = []
  rles = []
  for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

  import datetime

  sub = pd.DataFrame()
  sub['ImageId'] = new_test_ids
  sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
  sub.to_csv('../output/submission_'+datetime.datetime.now().strftime("%Y%m%d_%H%M")+'.csv', index=False)

