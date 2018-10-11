# Model construct with Keras APi with the aim to use 2 separate outputs (class and category) for future works
# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input, Concatenate, concatenate
from keras.regularizers import l2
import tensorflow as tf
import keras.backend as K

def get_axis():
	axis = -1 if K.image_data_format() == 'channels_last' else 1
	return axis


class TrafficSignsNet:

	@staticmethod
	def build_common_branch(inputs, chanDim=-1):
		weight_decay=1E-4	
		# (CONV => RELU) * 2 => POOL 
		# ---------------- 48 x 48 ------------------------		
		x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(inputs)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.2)(x)
		
		
		# (CONV => RELU) * 2 => POOL
		#----------------- 24 x 24 ------------------------
		x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.2)(x)
	
		# return the common layers output sub-network
		return x
	

	@staticmethod
	def build_class_branch(inputs, numClasses,
		finalAct="softmax", chanDim=-1):
		weight_decay=1E-4		
		# (CONV => RELU) * 2 => POOL
		# ---------------- 12 x 12 ---------------------
		x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(inputs)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		# ---------------- 6 x 6 ----------------------
		x = MaxPooling2D(pool_size=(2, 2))(x)		
		x = Dropout(0.35)(x)
		
		
		# --------------- 4608 = 128*6*6 ----------------		
		x = Flatten()(x)
		x = Dense(256, kernel_regularizer=l2(weight_decay))(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		x = Dropout(0.5)(x)
		x = Dense(numClasses, kernel_regularizer=l2(weight_decay))(x)
		x = Activation(finalAct, name="class_output")(x)

		# return the class prediction sub-network
		return x


	@staticmethod
	def build(width, height, numClasses, finalAct="softmax"):
		# initialize the input shape and channel dimension
		inputShape = (height, width, 3)
		chanDim = -1

		# construct the "class" net
		inputs = Input(shape=inputShape)
		commonInputs = TrafficSignsNet.build_common_branch(inputs, chanDim=chanDim)
		classBranch = TrafficSignsNet.build_class_branch(commonInputs, numClasses, finalAct=finalAct, chanDim=chanDim)

		# create the model using our input (the batch of images) and output 
		model = Model(
			inputs=inputs,
			outputs=classBranch,
			name="trafficSignsNet")

		# return the constructed network architecture
		return model
