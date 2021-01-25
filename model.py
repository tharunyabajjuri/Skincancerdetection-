# Import the required modules/dependencies
import os
import numpy as np
from keras.optimizers import rmsprop
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load the pretrained InceptionV3 model and discard the last layer
# Expect a Warning from Tensorflow
baseModel = InceptionV3(weights = 'imagenet', include_top = False)
print('Successfully loaded the InceptionV3 model!\n')

x = baseModel.output
x = GlobalAveragePooling2D()(x)
# Add the first dense layer
x = Dense(512, activation = 'relu')(x)
# Use Dropout to prevent overfitting
x = Dropout(0.5)(x)

# Add the final dense layer with softmax activation
predictions = Dense(2, activation = 'softmax')(x)

# Specify the inputs and the outputs and create the new model
myModel = Model(inputs = baseModel.input, outputs = predictions)

# Define the dictionary for Image Data Generation
dataGenArgs = dict(preprocessing_function = preprocess_input,
                  rotation_range = 30,
                  width_shift_range = 0.2,
                  height_shift_range = 0.2, 
                  shear_range = 0.2,
                  zoom_range = 0.2,
                  horizontal_flip = True,
                  vertical_flip = True)

trainDataGen = image.ImageDataGenerator(**dataGenArgs)
validDataGen = image.ImageDataGenerator(**dataGenArgs)

# Load the lesion images from the trainData directory
# Using binary as the class_mode
trainGenerator = trainDataGen.flow_from_directory(r'{}'.format('trainData'),
                                                 target_size = (299, 299),
                                                 color_mode = 'rgb',
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)

validGenerator = validDataGen.flow_from_directory(r'{}'.format('validationData'),
                                                 target_size = (299, 299),
                                                 color_mode = 'rgb',
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)

# Compile the model
myModel.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', 
                 metrics = ['accuracy'])

# Use the RMSProp Optimizer
# The loss function will be sparse categorical cross entropy
# The evaluation metric used will be accuracy

# Set the step size
stepSize = trainGenerator.n // trainGenerator.batch_size
validSteps = validGenerator.n // validGenerator.batch_size

# Train the model (use 20 epochs)
myModel.fit_generator(generator = trainGenerator, steps_per_epoch = stepSize, epochs = 20, 
                       validation_data = validGenerator, validation_steps = validSteps)                       
print('\nSucessfully trained the model!')

# Save the model for later use
myModel.save('model.h5')