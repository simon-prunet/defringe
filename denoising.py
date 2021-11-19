import numpy as np

from keras import layers
import warnings

from keras.preprocessing import image

from keras.models import Model
from keras.layers import Dense
from keras.layers import add
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K


def Xception_unet_2D(img_size):
	'''
		2D U-Net Xception denoiser
	'''
	inputs = Input(shape=img_size+(1,))

	### Encoder ###

	# Entry block
	x = Conv2D(32,3,strides=2,padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	# Keep residual
	previous = x

	for filters in [64, 128, 256]:

		x = SeparableConv2D(filters,3,padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)
		
		x = SeparableConv2D(filters,3,padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)

		x = MaxPooling2D(3, strides=2, padding="same")(x)

		# Now process residual and add it
		residual = Conv2D(filters,1,strides=2,padding="same")(previous)
		x = add([x,residual])
		previous = x

	### Decoder ###

	for filters in [256,128,64,32]:

		x = Conv2DTranspose(filters,3,padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)

		x = Conv2DTranspose(filters,3,padding="same")(x)
		x = BatchNormalization()(x)
		x = Activation("relu")(x)

		x = UpSampling2D(2)(x)

		residual = UpSampling2D(2)(previous)
		residual = Conv2D(filters,1,padding="same")(residual)
		x = add([x,residual])
		previous = x

	# Add final convolutive layer
	outputs = Conv2D(1,3,padding="same")(x)

	# Define model
	model = Model(inputs,outputs)
	return (model)



