# necessary imports
import numpy as np
import math
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, Activation, Dropout
from keras.callbacks import ModelCheckpoint

# Define data loaders for the datasets. 
# We have a mixed data types.
# The model will use data composed of a csv file and image files.
def data_loader (path) :
    # path directory to the driving_log csv dataset
    # which contains the recorded steering angle
    csv_path = path + 'driving_log.csv'
    # path directory to the image dataset
    img_folder_path = path + 'IMG/'
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            # Building the path for each dataset by using the split method
            # clr stands for center left right. clr_line is the data array (including path)
            # comprising the normalised images from the center, left, and right camera
            clr_line = line
            # Center camera image name.
            # We take the name at position 0 of the line array and split at the slash sign
            center_img_name = line[0].split('/')[-1]
            # Concatenate folder path and image name
            center_img_path = img_folder_path + center_img_name
            # The center image path is now filling the position 0 of the clr_line array
            clr_line[0] = center_img_path
            # We do the same for the left camera image name
            left_img_name = line[1].split('/')[-1]
            left_img_path = img_folder_path + left_img_name
            clr[1] = left_img_path # The left image path is now filling position 1 of the clr_line array
            # We do the same for the right camera image name
            right_img_name = line[2].split('/')[-1]
            right_img_path = img_folder_path + right_img_name
            clr_line[2] = right_img_path # The right image path is now filling position 2 of the clr_line array 
            # We append the clr_line array to the lines array.
            lines.append(clr_line)

# Define a data generator 
# Number of samples loaded for one batch = 128
def data_generator (data_lines, batch_size = 128) :
    num_lines = len(data_lines)
    while True:
        # shuffle the training data before each epoch
        shuffle(data_lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = data_lines[offset:offset + batch_size]
            images = []
            labels = []
            # Iterate and perform data augmentation/ normalization
            # We use the approach suggested in the instructions
            for batch_line in batch_lines:
                # Central camera image 
                center_img = cv2.cvtColor(cv2.imread(batch_line[0]), cv2.COLOR_BGR2RGB)
                images.append(center_img)
                center_label = float(batch_line[3])
                labels.append(center_label)
                # The central camera image is flipped 
                flipped_img = cv2.flip(center_img, flipCode=1)
                images.append(flipped_img)
                # The label of the central camera flipped image is changed  
                flipped_label = (-1.0)* center_label
                labels.append(flipped_label)
                
                # Images for two specific off-center shifts can be obtained 
                # from the left and the right camera
                # A correction factor is applied to compensate for the
                # change in point of view for both Left camera image and Right 
                # camera image. This was suggested in the instructions presentation.
                correction  = 0.2 # 0.2 was chosen from testing
                # Left camera image
                left_img = cv2.cvtColor(cv2.imread(batch_line[1]), cv2.COLOR_BGR2RGB)
                images.append(left_img)
                left_label  = center_label + correction
                labels.append(left_label)
                # The left camera image is flipped
                flipped_img = cv2.flip(left_img, flipCode=1)
                images.append(flipped_img)
                # The label sign of the left camera flipped image is changed 
                flipped_label = (-1.0)* left_label
                labels.append(flipped_label)
                # Right camera image
                right_img = cv2.cvtColor(cv2.imread(batch_line[2]), cv2.COLOR_BGR2RGB)
                images.append(right_img)
                right_label = center_label - correction
                labels.append(right_label)
                # The right camera image is flipped
                flipped_img = cv2.flip(right_img, flipCode=1)
                images.append(flipped_img)
                # The label of the right camera flipped image is changed
                flipped_label = (-1.0)* right_label
                labels.append(flipped_label)

            X_train = np.array(images)
            y_train = np.array(labels)
            # In a generator function, you  use the yield keyword to 
            # perform iteration inside a while True: loop. Each time Keras 
            # calls the generator, it gets a batch of data and it automatically 
            # wraps around the end of the data.
            yield shuffle(X_train, y_train)

# Initialize our lines array 
# It comprises all the data after augmentation 
lines = []    
# Path of the Udacity data. 
# The data is accessible only in GPU enabled mode and is located
# in the directory above the home directory.
path = '../../../opt/carnd_p3/data/'
data_loader(path)
# Printing the size of the Udacity dataset.
# Dataset is not that big but it is enough to see if the model is good for a start.
print('len lines= ',len(lines))
# Keep 20% of dataset for validation
train_data, val_data = train_test_split(lines, test_size = 0.2)

# Nvidia pipeline
# As suggested in the instructions, I followed the NVidia pipeline architecture
# The paper describing it is available at https://arxiv.org/pdf/1604.07316v1.pdf
model = Sequential()
# As suggested in the instructions, we use a cropping layer that removes
# the top and bottom parts of the images
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# As suggested in the instructions, we use a lambda layer to normalize the data
model.add(Lambda (lambda x: (x / 255.0) - 0.5) )
# 5 Convolutional layers folled by an Exponential Linear Unit activation function
# ELUs alleviate the vanishing gradient problem via the identity for positive values.
# paper available at https://arxiv.org/abs/1511.07289
# Convolutional layer, dimensionality of the output space 24, 5×5 kernel , 2x2 stride, Exponential Linear Unit activation function
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
# Convolutional layer, dimensionality of the output space 36, 5×5 kernel , 2x2 stride, Exponential Linear Unit activation function
model.add(Conv2D(36, (5,5), strides=(2, 2), activation="elu"))
# Convolutional layer, dimensionality of the output space 48, 5×5 kernel , 2x2 stride, Exponential Linear Unit activation function
model.add(Conv2D(48, (5,5), strides=(2, 2), activation="elu"))
# Convolutional layer,dimensionality of the output space 64, 3x3 kernel, 1x1 stride, Exponential Linear Unit activation function
model.add(Conv2D(64, (3,3), activation="elu"))
# Convolutional layer, dimensionality of the output space 64, 3x3 kernel, 1x1 stride, Exponential Linear Unit activation function
model.add(Conv2D(64, (3,3), activation="elu"))
# Flatten image
model.add(Flatten())
# Keras does automatic shape inference
# Fully connected layer, Exponential Linear Unit activation function
model.add(Dense(100, activation="elu"))
# Dropout layer to avoid overfitting
model.add(Dropout(0.25))
# Fully connected layer, Exponential Linear Unit activation function
model.add(Dense(50, activation="elu"))
# Fully connected layer, Exponential Linear Unit activation function
model.add(Dense(10, activation="elu"))
# Fully connected layer
model.add(Dense(1))
# Compile the model. Use MSE and Adam optimizer
model.compile(loss='mse',optimizer='adam')
# Train the model
model.fit_generator(generator = data_generator(train_data),
                    validation_data = data_generator(val_data),
                    epochs = 2, # More epochs training does not really decrease the mse.
                    steps_per_epoch  = math.ceil(len(train_data) / 128),
                    validation_steps = math.ceil(len(val_data)   / 128)    )
model.save('model.h5')
print('Model saved')