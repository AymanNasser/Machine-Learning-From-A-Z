import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

# Image Augmentaion: Transforming images of the training set to avoid over-fitting
# Training Set
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train_datagen.flow_from_directory('training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Test Set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Building CNN
cnn = tf.keras.models.Sequential()

# Convolution Layer
# input_shape: shaping our images as 64*64 pixel of RGB so it is 64*64*3
# kernel_size: n*n matrix used for kernel for image convolution
cnn.add(tf.keras.layers.Conv2D(input_shape=[64,64,3],filters= 32, kernel_size= 3, activation= 'relu'))

# Pooling Layer
# Sliding n*n pixels to get the pooling matrix out of the feature map
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2, padding='valid'))

# Adding 2nd convolution layer
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation= 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2, padding='valid'))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connected Layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the cnn
cnn.compile(metrics=['accuracy'], optimizer='adam', loss='binary_crossentropy')
cnn.fit(x=training_set, validation_data= test_set, epochs=15)

# Making a single prediction
test_img = image.load_img('/home/an/FOEASU-Linux/Machine Learning/Datasets/Cats_Dogs/single_prediction/cat_or_dog_1.jpg',
                          target_size=(64,64))
# Converting PIL image format to 2d array
test_img = image.img_to_array(test_img)
# Adding a batch dimension to the image as the first dimension: axis=0
test_img = np.expand_dims(test_img, axis=0)

result = cnn.predict(test_img)
# 1 Corresponds to dog
# Accesing the batch as result[0] then accessing the single element of the batch result[0][0]
if result[0][0] == 1:
        print('Dog')
else:
        print('Cat')



