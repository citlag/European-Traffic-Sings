# Needs keras version 1, with keras 2 does not work -> Merge option to merge 2 sequential models
# NOTE -> It worked with keras 2.0.8 and Tensorflow 1.10.1
import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
import os
import glob
import h5py
from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l2


from keras import backend as K
import numpy as np


from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in y
	hsv = color.rgb2hsv(img)
	hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
	img = color.hsv2rgb(hsv)

	# rescale to standard size  
	img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

	return img


def get_class(img_path):
    return int(img_path.split('/')[-2])



#------- Read training images---------------------------------------------------
from skimage import io
import os
import glob


def get_class(img_path):
    return int(img_path.split('/')[-2])


try:
    with  h5py.File('X_train_german.h5') as hf: 
        X, Y = hf['imgs'][:], hf['labels'][:]
    print("Loaded images from X_train_german.h5")
    
except (IOError,OSError, KeyError):  
    print("Error in reading X_train_german.h5. Processing all images...")
    root_dir = '../../Datasets/Traffic_signs/German_Recognition/Final_Training/Images'
    imgs = []
    labels = []

    # Read all image paths with extension ppm
    all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*/*.ppm')) )
    np.random.seed(42)
    np.random.shuffle(all_img_paths)
    
    # Read images and preprocess them
    print("[INFO] loading images...")
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))            
            label = get_class(img_path)            
            
            # Save images and labels in lists
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    # Save the training dictionary
    with h5py.File('X_train_german.h5','w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)


#--------------- Read test images-------------------------------
try:
    with  h5py.File('X_test_german.h5') as hf: 
        X_test, y_test = hf['imgs'][:], hf['labels'][:]
    print("Loaded images from X_test_german.h5")
    
except (IOError,OSError, KeyError):  
    print("Error in reading X_test_german.h5. Processing all images...")
    root_dir = '../../Datasets/Traffic_signs/German_Recognition/Final_Test/Images2'
    X_test = []
    y_test = []

    # Read all image paths with extension ppm
    all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*/*.ppm')) )
    np.random.seed(42)
    np.random.shuffle(all_img_paths)

    # Read images and preprocess them    
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))            
            label = get_class(img_path)
            
            # Save images and labels in lists
            X_test.append(img)
            y_test.append(label)

            if len(X_test)%1000 == 0: print("Processed {}/{}".format(len(X_test), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='uint8')
    # Save testing dictionary
    with h5py.File('X_test_german.h5','w') as hf:
        hf.create_dataset('imgs', data=X_test)
        hf.create_dataset('labels', data=y_test)


#------------- Split data into train and validation-------------------
# Read data and split it in train and validation
from keras import utils as np_utils
from sklearn.cross_validation import train_test_split

# Split data into train and validation
# random_state helps defining the same split always. If you do not define it, the split will always be different
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

normalize = 0
# Normalize the data: subtract the mean image
if normalize:
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image


print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# ----------------- Initialization of weights -------------------------------
from keras import backend as K
import numpy as np

def my_init(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)


# --------------- Define Keras model ---------------------------------- 
# 注意keras使用tensorflow和thano不同后台， 数据输入的通道顺序不同哦
def cnn_model():
    branch_0= Sequential()
    branch_1 = Sequential()
    model0 = Sequential()
    model = Sequential()
    # ********************************************** 48*48
    model0.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal' , input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model0.add(BatchNormalization(epsilon=1e-06, axis=3))
    model0.add(Activation('relu'))
  
    model0.add(Convolution2D(48, 7, 1, border_mode='same', init='he_normal'))
    model0.add(BatchNormalization(epsilon=1e-06, axis=3))
    model0.add(Activation('relu'))
    model0.add(Convolution2D(48, 1, 7, border_mode='same', init='he_normal'))
    model0.add(BatchNormalization(epsilon=1e-06, axis=3))
    model0.add(Activation('relu'))   
    model0.add(MaxPooling2D(pool_size=(2, 2)))
    model0.add(Dropout(0.2))
    # ****************************************** 24*24
    branch_0.add(model0)
    branch_1.add(model0)
    
    branch_0.add(Convolution2D(64, 3, 1, border_mode='same', init='he_normal'))
    branch_0.add(BatchNormalization(epsilon=1e-06, axis=3))
    branch_0.add(Activation('relu'))
    branch_0.add(Convolution2D(64, 1, 3, border_mode='same', init='he_normal'))
    branch_0.add(BatchNormalization(epsilon=1e-06,  axis=3))
    branch_0.add(Activation('relu'))
    
    branch_1.add(Convolution2D(64, 1, 7, border_mode='same', init='he_normal'))
    branch_1.add(BatchNormalization(epsilon=1e-06, axis=3))
    branch_1.add(Activation('relu'))
    branch_1.add(Convolution2D(64, 7, 1, border_mode='same', init='he_normal'))
    branch_1.add(BatchNormalization(epsilon=1e-06, axis=3))
    branch_1.add(Activation('relu'))    
    
    model.add(Merge([branch_0, branch_1], mode='concat', concat_axis=-1))   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # ******************************************* 12*12
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization(epsilon=1e-06, axis=3))
    
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', init='he_normal'))   # 之前是256个滤波器
    model.add(BatchNormalization(epsilon=1e-06, axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # *************************************** 6*6
    model.add(Flatten())
    model.add(Dense(256, init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax', init='he_normal'))
    return model


# ----------- Initialize the model ----------------
model = cnn_model() 
model.summary() # Print model information

# let's train the model using SGD + momentum (how original).
lr = 0.001
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
adm = Adam(lr=0.001, decay=1e-6)  #之前没有设置decay
model.compile(loss='categorical_crossentropy',
              optimizer=adm,
              metrics=['accuracy'])


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch/10))


# ------------- Plot and save model ----------------
#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')



# ------------- Start training --------------------
batch_size = 128
nb_epoch = 40
t1=time()

history = model.fit(X_train, y_train, batch_size=batch_size, 
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, y_val),
                            shuffle = True
         )
t2=time()
print ('Time taken to train the model:', t2-t1)

model.save('weights_german2/german_epoch40.h5')

# ----------------- Test on test dataset --------------------
t1=time()
y_pred = model.predict_classes(X_test)
t2=time()
acc = np.mean(y_pred==y_test)
print("Test accuracy = {}".format(acc	))
print  ('{} sec to predict {} images on the test set'.format(t2-t1, len(X_test)))


# ---------------- Plot the training history ----------------
# Loss
plt.figure(figsize=(6,5)) 
plt.plot(history.history['loss'], '-o')
plt.plot(history.history['val_loss'], '-x')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show()

# Accuracy
plt.figure(figsize=(6,5)) 
plt.plot(history.history['acc'], '-o')
plt.plot(history.history['val_acc'], '-x')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc='lower right')
plt.show()





#### +++++++++++++++++++++ WITH DATA AGUMENTATION ++++++++++++++++++++++++++++++++++
from sklearn.cross_validation import train_test_split

datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)


# ----------- Training based on the former model -------------------
'''
model2 = cnn_model() 

model2.compile(loss='categorical_crossentropy',
              optimizer=adm,
              metrics=['accuracy'])
'''
#model.load_weights('weights_german2/german_epoch40.h5') # Used as weight initialization if desired

nb_epoch = 50
batch_size = 128
t1=time()
history2=model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, y_val),
                            callbacks=[ReduceLROnPlateau('val_loss', factor=0.2, patience=20, verbose=1, mode='auto'), 
                                       ModelCheckpoint('weights_german2/german_aug_best50.h5',save_best_only=True)]
                           )
t2=time()
print ('Time taken to train the augmented model:', t2-t1)

model.save('weights_german2/german_aug_epoch50.h5')


# ---------------- Plot the training history ----------------
# Loss
plt.figure(figsize=(6,5)) 
plt.plot(history2.history['loss'], '-o')
plt.plot(history2.history['val_loss'], '-x')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show()

# Accuracy
plt.figure(figsize=(6,5)) 
plt.plot(history2.history['acc'], '-o')
plt.plot(history2.history['val_acc'], '-x')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc='lower right')
plt.show()

# -------------- Test trained model with data augmentation on test set ---------------------
t1=time()
y_pred = model.predict_classes(X_test)
t2=time()
acc = np.mean(y_pred==y_test)
print("Test accuracy = {}".format(acc))
print  ('{} sec to predict {} images on the test set'.format(t2-t1, len(X_test)))



