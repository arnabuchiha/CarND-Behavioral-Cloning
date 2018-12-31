
# coding: utf-8

# In[39]:


import os
import csv
from matplotlib import pyplot as plt

samples=[]
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0)
# In[40]:


from sklearn.model_selection import train_test_split
train_samples,validation_samples=train_test_split(samples,test_size=0.2)


# In[41]:


import cv2,os
import numpy as np
import sklearn
from scipy.misc import imread
from sklearn.utils import shuffle

# In[ ]:





# In[47]:
data_dir="/opt/carnd_p3/data/"

def generator(samples,batch_size=32):
    num_samples=len(samples)
    correction=0.25
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
            images=[]
            angles=[]
            for batch_sample in batch_samples:
#                 name="/opt/carnd_p3/data/IMG/"+batch_sample[0].split('/')[-1]
#                 left_name="/opt/carnd_p3/data/IMG/"+batch_sample[1].split('/')[-1]
#                 right_name="/opt/carnd_p3/data/IMG/"+batch_sample[2].split('/')[-1]
                left_image=imread(data_dir+batch_sample[1].strip())
                right_image=imread(data_dir+batch_sample[2].strip())
                center_image=imread(data_dir+batch_sample[0].strip())
#                 left_image=cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
#                 right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
#                 center_image=cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                center_angle=float(batch_sample[3])
                
                left_angle=center_angle+correction
                right_angle=center_angle-correction
                
                image_flipped = cv2.flip(center_image, 1)
                measurement_flipped = -center_angle
                
                
                images.append(center_image)
                angles.append(center_angle)
                
                images.append(image_flipped)
                angles.append(measurement_flipped)
                            
                
                images.append(left_image)
                angles.append(left_angle)
                
                images.append(right_image)
                angles.append(right_angle)
            
            X_train=np.array(images)
            y_train=np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)


# In[48]:


train_generator=generator(train_samples,batch_size=32)
validation_generator=generator(validation_samples,batch_size=32)


# In[49]:


from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Cropping2D,BatchNormalization,Activation,Dropout,MaxPooling2D,Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[50]:


# ACTIVATION='elu'
# model = Sequential()
# model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
# model.add(Cropping2D(cropping=((74,24), (0,0))))
# model.add(Conv2D(24,5,5,subsample=(2,2),activation=ACTIVATION))
# model.add(Conv2D(36,5,5,subsample=(2,2),activation=ACTIVATION))
# model.add(Conv2D(48,5,5,subsample=(2,2),activation=ACTIVATION))
# model.add(Conv2D(64,3,3,activation=ACTIVATION))
# model.add(Conv2D(64,3,3,activation=ACTIVATION))
# model.add(Flatten())
# model.add(Dropout(.5))
# model.add(Dense(1164))
# model.add(Activation(ACTIVATION))
# model.add(Dense(100))
# model.add(Activation(ACTIVATION))
# model.add(Dense(50))
# model.add(Activation(ACTIVATION))
# model.add(Dense(10))
# model.add(Activation(ACTIVATION))
# model.add(Dense(1))
# model.summary()
model=Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2),W_regularizer=l2(0.001)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2),W_regularizer=l2(0.001)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2),W_regularizer=l2(0.001)))
model.add(Conv2D(64, 3, 3, activation='elu',W_regularizer=l2(0.001)))
model.add(Conv2D(64, 3, 3, activation='elu',W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu',W_regularizer=l2(0.001)))
model.add(Dense(50, activation='elu',W_regularizer=l2(0.001)))
model.add(Dense(10, activation='elu',W_regularizer=l2(0.001)))
model.add(Dense(1))
model.summary()


# In[51]:


# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)
checkpointer = ModelCheckpoint(filepath='model-{val_loss:.5f}.h5', verbose=1, save_best_only=True)
model.compile(loss='mse', optimizer=Adam(lr=1e-4))
fit_loss = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=5,callbacks=[checkpointer],verbose=1)


# In[ ]:


plt.plot(fit_loss.history['loss'])
plt.plot(fit_loss.history['val_loss'])
plt.title('Mean Squared Error Loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:


model.save('model.h5')

