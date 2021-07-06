from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.applications.resnet
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
# tf.keras.backend.clear_session()

AUTOTUNE = tf.data.experimental.AUTOTUNE


batch_size = 32

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         horizontal_flip=True,
#         vertical_flip=True,
#         rotation_range = 15)

train_datagen = tf.keras.Sequential([
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
        label_mode="categorical",
        
        # save_to_dir= './newDataset/'
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
# validation_generator = validation_generator.map(lambda x, y: (test_datagen(x, training=True), y))
validation_generator = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)



model = tensorflow.keras.applications.resnet.ResNet50(include_top=True,input_shape=(200,200,3),weights=None,classes=16)
# Freeze the layers except the last 4 layers
# for layer in res_model.layers[:-4]:
#     layer.trainable = False
 
# # Check the trainable status of the individual layers
# for layer in res_model.layers:
#     print(layer, layer.trainable)
    
# model = models.Sequential()
 
# # Add the vgg convolutional base model
# model.add(res_model)
 
# # Add new layers
# model.add(layers.Flatten())
# model.add(layers.Dense(2048, activation='relu'))
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(16, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()
op = Adam(lr=0.001)    
model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
mc = ModelCheckpoint('./modelos/Resnet50best_modelacc2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
mc2 = ModelCheckpoint('./modelos/Resnet50best_modelloss2.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


history = model.fit(
        train_generator,
        epochs=1000,
        validation_data=validation_generator,
        callbacks=[es,mc,mc2, tensorboard_callback]
)