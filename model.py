import csv
import cv2
import math
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout,Convolution2D,Cropping2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


#parsing csv file
lines = []

'''
csv_file = open("data/driving_log.csv")
reader = csv.reader(csv_file)

[lines.append(x) for x in reader]
'''

[lines.append(x) for x in csv.reader(open("data/driving_log.csv"))]
#parsing inputs from the csv files

images = []
measurements = []
del lines[0]


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

reaction_distance = 7
camera_offest = 0.9
delta_theta = math.atan(reaction_distance / camera_offest)
delta_theta = 0.2

'''
for tmp in lines:

	img_path = tmp[0]
	lft_img  = tmp[1]
	rgt_img  = tmp[2]
	angle    = float(tmp[3])

	#adding center image
	c_img = cv2.imread("data/" + img_path)
	
	images.append(c_img)
	measurements.append(angle)
	
	images.append(np.fliplr(c_img))
	measurements.append(-angle)

	#adding left image
	l_img   = cv2.imread("data/" + lft_img[1:])
	l_angle = angle + delta_theta
	
	images.append(l_img)
	measurements.append(l_angle)

	images.append(np.fliplr(l_img))
	measurements.append(-l_angle)
	
	#adding right image
	r_img   = cv2.imread("data/" + rgt_img[1:])
	r_angle = angle - delta_theta

	images.append(r_img)
	measurements.append(r_angle)

	images.append(np.fliplr(r_img))
	measurements.append(-r_angle)

x_train = np.array(images)
y_train = np.array(measurements)
'''

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		samples = sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

		images = []
		angles = []
		for batch_sample in batch_samples:
			#center image and its flipped version
			name = 'data/IMG/'+batch_sample[0].split('/')[-1]
			center_image = cv2.imread(name)
			center_angle = float(batch_sample[3])
                
			images.append(center_image)
			angles.append(center_angle)

			images.append(np.fliplr(center_image))
			angles.append(-center_angle)

			#left image and its flipped version
			l_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
			left_image = cv2.imread(l_name)
			left_angle = center_angle + delta_theta

			images.append(left_image)
			angles.append(left_angle)

			images.append(np.fliplr(left_image))
			angles.append(-left_angle)

			#right image and its flipped version
			r_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
			right_image = cv2.imread(r_name)
			right_angle = center_angle - delta_theta

			images.append(right_image)
			angles.append(right_angle)

			images.append(np.fliplr(right_image))
			angles.append(-right_angle)

		# trim image to only see section with road
		X_train = np.array(images)
		y_train = np.array(np.asarray(angles))
		yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32*6)
validation_generator = generator(validation_samples, batch_size=32*6)

model = Sequential()

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#using nvidia model https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(1164 , activation='relu'))
model.add(Dropout(0.65))
model.add(Dense(100 , activation='relu'))
model.add(Dense(50 , activation='relu'))
model.add(Dense(10 , activation='relu'))
model.add(Dense(1))

model.summary()
callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

model.compile(loss="mse" , optimizer = "adam")

#model.fit(x_train , y_train , nb_epoch=10, validation_split = 0.2 , shuffle = True , callbacks=[callback] )
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=30 ,callbacks=[callback])

model.save("model.h5")

