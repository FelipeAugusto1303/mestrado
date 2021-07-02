# -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D, Input,add, Add, Flatten, Dropout, BatchNormalization, Activation
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt 
# from livelossplot import PlotLossesKeras

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

AUTOTUNE = tf.data.experimental.AUTOTUNE


batch_size = 64

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         horizontal_flip=True,
#         vertical_flip=True,
#         rotation_range = 15)

train_datagen = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.30),
    tf.keras.layers.experimental.preprocessing.RandomFlip()
    ])

# test_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])

train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        './dataset/train_pose/',
        image_size=(200, 200),
        color_mode = 'rgb',
        batch_size=batch_size,
        label_mode="categorical"
        
        #save_to_dir= 'dataset/',
        #subset='training',
        )
train_generator = train_generator.map(lambda x, y: (train_datagen(x, training=True), y))
train_generator = train_generator.cache().prefetch(buffer_size=AUTOTUNE)


validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
        './dataset/test_pose/',
        image_size=(200, 200),
        color_mode = 'rgb',
        batch_size=batch_size,
        label_mode="categorical"
        #shuffle= 'True',
        #subset='validation',
        )
validation_generator = validation_generator.map(lambda x, y: (test_datagen(x, training=True), y))
validation_generator = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)

inp = Input(shape = (200,200,3))


x = Conv2D(96, kernel_size=(11,11),strides=4, activation='relu',padding='same')(inp)
x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
x = Conv2D(256, kernel_size=(5,5), activation='relu',padding='same')(x)
x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
x = Conv2D(384, kernel_size=(3,3), activation='relu',padding='same')(x)
x = Conv2D(384, kernel_size=(3,3), activation='relu',padding='same')(x)
x = Conv2D(256, kernel_size=(3,3), activation='relu',padding='same')(x)
x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
# x = Conv2D(128, kernel_size=(3,3),strides=(2,2), activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(0.01))(x)

# x = MaxPooling2D(pool_size=(3,3))

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)

predictions = Dense(16, activation = 'softmax', name='camada_saida')(x)
model = Model(inputs=inp, outputs=predictions)

op = SGD(learning_rate=0.001, momentum=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = op, metrics = ['accuracy'])
es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=10)
mc = ModelCheckpoint('./modelos/alexNet2_acc.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
mc2 = ModelCheckpoint('./modelos/alexNet2_loss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model.summary()

history = model.fit(
        train_generator,
        epochs=1000,
        validation_data=validation_generator,
        callbacks=[es,mc,mc2, tensorboard_callback]
)

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()