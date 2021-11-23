import numpy as np

from tensorflow.keras import layers
import warnings

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam 


def denoising_unet_2D(img_size, start_neurons=16,dropout=0.5):
	'''
		2D U-Net denoiser
	'''
	inputs = Input(shape=img_size+(1,))

	### Encoder ###


	conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D((2,2))(conv1)
	pool1 = Dropout(dropout)(pool1)

	conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D((2,2))(conv2)
	pool2 = Dropout(dropout)(pool2)

	conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D((2,2))(conv3)
	pool3 = Dropout(dropout)(pool3)

	conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D((2,2))(conv4)
	pool4 = Dropout(dropout)(pool4)

  ### Middle ####

	convm = Conv2D(start_neurons*16, (3,3), activation='relu', padding='same')(pool4)
	convm = Conv2D(start_neurons*16, (3,3), activation='relu', padding='same')(convm)


	### Decoder ###
	deconv4 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(convm)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
	uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
	uconv4 = Dropout(dropout)(uconv4)

	deconv3 = Conv2DTranspose(start_neurons*4,(3,3), strides=(2,2), padding='same')(uconv4)
	uconv3 = concatenate([deconv3, conv3])
	uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
	uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
	uconv3 = Dropout(dropout)(uconv3)

	deconv2 = Conv2DTranspose(start_neurons*2,(3,3), strides=(2,2), padding='same')(uconv3)
	uconv2 = concatenate([deconv2, conv2])
	uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
	uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
	uconv2 = Dropout(dropout)(uconv2)

	deconv1 = Conv2DTranspose(start_neurons*1,(3,3), strides=(2,2), padding='same')(uconv2)
	uconv1 = concatenate([deconv1, conv1])
	uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
	uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
	uconv1 = Dropout(dropout)(uconv1)

	outputs = Conv2D(1, (1,1), padding='same')(uconv1)

	# Define model
	model = Model(inputs,outputs)
	return (model)



def test_denoise(start_neurons=16,dropout=0.25,learning_rate=0.005,epochs=15):

        '''
        Small driver routine to train and test u-net
        '''
        # Generate data set, add noise, split in train/test sets
        from synthetic_data import generate_synth_data
        fim,bim,msk,seqind = generate_synth_data()
        truth = fim-bim
        noisy = truth + 10.*np.random.randn(*truth.shape)
        truth = np.reshape(truth,truth.shape+(1,))
        noisy = np.reshape(noisy,noisy.shape+(1,))
        from sklearn.model_selection import train_test_split
        noisy_train, noisy_val, truth_train, truth_val = train_test_split(noisy, truth,test_size=0.2)
        opt = Adam(learning_rate=learning_rate)
        m = denoising_unet_2D((80,80),start_neurons=start_neurons,dropout=dropout)
        m.compile(loss="mse",optimizer=opt)
        m.fit(noisy_train,truth_train,batch_size=128,epochs=epochs,validation_data=(noisy_val,truth_val))
        res = m.predict(noisy_val)
        return truth_val, noisy_val, res


        
