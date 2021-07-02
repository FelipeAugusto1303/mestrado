import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
# from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras import optimizers

%matplotlib inline

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

batch_size = 64

train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip = True,
        rotation_range = 15)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './dataset/train_pose/',
        target_size=(200, 200),
        color_mode = 'rgb',
        batch_size=batch_size,
        class_mode='categorical',
        #save_to_dir= 'dataset/',
        #subset='training',
        )

validation_generator = test_datagen.flow_from_directory(
        './dataset/test_pose/',
        target_size=(200, 200),
        color_mode = 'rgb',
        batch_size=batch_size,
        class_mode='categorical',
        #shuffle= 'True',
        #subset='validation',
        )

vgg_model = VGG16(include_top=False,input_shape=(200,200,3),weights=None,classes=16)
vgg_output = vgg_model.output
x = Flatten()(vgg_output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.15)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.15)(x)

predictions = Dense(16, activation = 'softmax', name='camada_saida')(x)
model = Model(inputs=vgg_model.input, outputs=predictions)


model.summary()
op = Adam(lr=0.0001)    
model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=5)
mc = ModelCheckpoint('./modelos/VGG16_modelacc.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
mc2 = ModelCheckpoint('./modelos/VGG16_modelloss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=39423/batch_size,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=10043/batch_size,
        callbacks=[es,mc,mc2, tensorboard_callback])

