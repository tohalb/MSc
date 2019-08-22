
# coding: utf-8

# In[250]:
# Based on article by Dong et al: Image Super-Resolution Using Deep Convolutional Networks

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy
import pdb

#Extra imports added here
import imageio
from skimage import measure


# In[251]:


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


# In[252]:


def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image


# In[253]:


def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_


# In[254]:


"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255


# In[255]:


"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5

#Note: truncated_normal does not seem to have any beneficial effect
initializer = tf.random_normal
weights = {
    'w1': tf.Variable(initializer([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(initializer([1, 1, 64,32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(initializer([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }
"""Define the model layers with three convolutional layers
"""


# conv1 layer with biases and relu : 64 filters with size 9 x 9
#  Stride = 1

# Note: Valid padding seems to require cropping at the psnr comparison stage - possibly introducing artefacts
conv1 = tf.nn.relu(tf.nn.conv2d(inputs, filter=weights['w1'], strides=[1, 1, 1, 1], padding='VALID', name='conv1') + biases['b1'])
##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, filter=weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, filter=weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b3']


# In[256]:


"""Load the pre-trained model file
"""
model_path='model\\model.npy'
model = np.load(model_path, encoding='latin1').item()


# In[257]:


## show the weights of model and visualisa
# variabiles (w1, w2, w3)

def visualise_w1():
    fig = plt.figure(figsize=(8, 8))  # width, height in inches
    for i in range(64):
        sub = fig.add_subplot(8, 8, i + 1)
        sub.imshow(weights1[:, :, 0, i], cmap='gray')
        plt.axis('off')
    print('Visualise Weights w1:')
    plt.show()

def visualise_w2():
    #Need to set vmin and vmax to prevent imshow rescaling 1x1 relative to all pixels i.e. black
    minval = weights2.min()    
    maxval = weights2.max()
    
    if abs(maxval) > abs(minval):
        minval = maxval * -1
    else:
        maxval = minval * -1

    fig = plt.figure(figsize=(8,8))  # width, height in inches
    for i in range(32):
        sub = fig.add_subplot(4, 8, i + 1)
        sub.imshow(weights2[:, :, 0, i],cmap='gray', vmin = minval, vmax=maxval)
        plt.axis('off')
    print('Visualise Weights w2:')
    plt.show()


def visualise_w3():
    fig = plt.figure(figsize=(2, 2))
    plt.imshow(weights3[:, :, 0, 0], cmap='gray')
    plt.axis('off')
    print('Weights w3:')
    plt.show()



# In[258]:


#Show weights
weights1 = model['w1']
weights2 = model['w2']
weights3 = model['w3']


print(weights1.min())

#Show Values for weights
print('Show Weights w1:')
print(weights1)
print('Show Weights w2:')
print(weights2)
print('Show Weights w3:')
print(weights3)




# In[259]:


#Visualise weights

visualise_w1()

visualise_w2()

visualise_w3()


# In[260]:


"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
# Note: preprocess function normalises and applies bicubic interpolation
# At this stage images are therefore the same
blurred_image, groundtruth_image = preprocess('image\\butterfly_GT.bmp')


"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# Note: Only using output layer as pretrained model provided

output_ = sess.run(conv3, feed_dict={inputs: input_})


# In[261]:


#save the blurred and SR images and compute the psnr
# use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'

#Ground truth grayscale image - has only been through preprocessing
ground = imageio.imwrite('grnd.jpg', groundtruth_image)

#Image after interpolation
blurred = imageio.imwrite('blur.jpg', blurred_image)

# Image after tensorflow session run
# Note: Warning about lossy conversion - float64 has huge range, graphic uint 8 is only 0 - 255
output_float = output_.astype(np.float64)
output_img = imageio.imwrite('hr_output.jpg',output_float[0,:,:,0])

output_float = output_.astype(np.float64)

# Note: Resizing because VALID padding causes mismatching dimensionality.
# Using 'SAME' padding makes this unnecessary but results in lower psnr
hr_psnr = measure.compare_psnr(groundtruth_image[6:249,6:249], output_float[0, :, :, 0])

bicubic_psnr = measure.compare_psnr(groundtruth_image, blurred_image)

print('Super Resolution PSNR:      ', hr_psnr)
print('Bicubic Interpolation PSNR: ', bicubic_psnr)

