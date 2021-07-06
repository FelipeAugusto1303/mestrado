import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt

# classes = []
# for path in glob.glob('./dataset/train_pose/*'):
#     classes.append(path.split('/')[-1].split('\\')[-1])

# print(classes)

batch_size = 128

train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        './dataset/train_pose/',
        image_size=(200, 200),
        color_mode = 'rgb',
        batch_size=batch_size,
        label_mode="categorical",
        
        # save_to_dir= './newDataset/'
        #subset='training',
        )

train_datagen = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.30),
    tf.keras.layers.experimental.preprocessing.RandomFlip(),
    tf.keras.layers.experimental.preprocessing.RandomZoom((0.1,0.9))
    ])

classes = train_generator.class_names
print(classes)

plt.figure(figsize=(10,10))

for image, labels in train_generator.take(1):
    for i in range(9):
        augmented_images = train_datagen(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
        plt.savefig("dataaugmentation.png")