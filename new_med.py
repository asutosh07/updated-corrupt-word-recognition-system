import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

NAME="project_training_with_convLayers=(32,64)dropout=(0.2,0.2),denseLayers=(64,128,256,512,1024)dropout=(0.3,0.4,0.4,0.5,0.5),10 epochs"

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#Adding a Dense Layer of 512 units and dopout of value 0.5
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5))

#Adding a second Layer of 256 activation of value of 0.5
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 5, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

#Processing Train Set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)
#Processing Test set
test_datagen = ImageDataGenerator(rescale = 1./255)





#Generating Train Set
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 subset='training',
                                                 shuffle=True)
#Generating Test Set
test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64,64),
                                            batch_size = 1,
                                            class_mode = 'categorical',
                                            shuffle=True)
#Keeping logs of accuracy and losses for every different runs using tensorboard
tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

#Assining Parameters to model
classifier.fit_generator(training_set,
                         steps_per_epoch =218,
                         epochs =5,
                         validation_data = test_set,
                         validation_steps =20,
                         callbacks=[tensorboard]
                         )


#Importing some image which are not trained before and checking how correctly our model predicts those
class_names = ['Histafree','P250','phenzee','rhinoclear','zinova']
from keras.preprocessing import image
import matplotlib.pyplot as plt


test_image = image.load_img('rhinoclear_test.jpg', target_size = (64,64))
plt.grid(False)
plt.imshow(test_image)
plt.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)
print('Actual:  '+class_names[3])
print('Predicted:  '+class_names[np.argmax(result)])
