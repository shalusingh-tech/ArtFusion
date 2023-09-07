
# setup librarys
import functools
import os
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# function to crop image from center
def crop_center(image):
  '''Return croped square image.'''
  shape = image.shape
  #print('Image Shape:',shape)
  new_shape = min(shape[1],shape[2])
  offset_y = max(shape[1] - shape[2],0)//2  # height
  offset_x = max(shape[2] - shape[1],0)//2  # width
  image = tf.image.crop_to_bounding_box(image,offset_y,offset_x,new_shape,new_shape )
  return image


# function: show  images
# def show_images(images,titles =('',)):
#   n = len(images)
#   images_shape = [image.shape[1] for image in images]
#   w = (images_shape[0] * 6)//320
#   plt.figure(figsize=(w*n,w))
#   gs = gridspec.GridSpec(1,n,width_ratios = images_shape)
#   for i in range(n):
#     plt.subplot(gs[i])
#     plt.imshow(images[i][0],aspect = 'equal')
#     plt.axis('off')
#     plt.title(titles[i] if len(titles) > i else '' )
#   plt.show()

# function: image info
def img_info(img):
  print('Shape:',img.shape)
  print('Min. Value:',img.min())
  print('Max. Value:',img.max())
  print('Data type:',img.dtype)
  print('Image size:%f MB'%(img.nbytes/1024**2))


# function: load and preprocess image from local storage
def load_local_img(img_path,img_size = (720,720),perserve_aspect_ratio = True):
  # load the image from local machine
  try:
    img = Image.open(img_path)
  except:
    print('Image Failed to load')
  img = np.asarray(img)  # convert to numpy array
  img = np.expand_dims(img,axis = 0) # new img_shape: [batch,height,width,channel]
  # crop the image to center
  #img = crop_center(img)
  # resize the image to desired size
  img = tf.image.resize(img,img_size,preserve_aspect_ratio = True)
  img = img.numpy() # convert to tensor to numpy
  img = img/255 # pixel value in range [0,1]
  return img

# function: resize image to desired size
def resize_img(img,image_size=(360,360)):
  img = tf.convert_to_tensor(img)
  img = tf.image.resize(img,image_size,preserve_aspect_ratio=True)
  return img

# function: to load the trained image
def get_model(path):
  dif_model = tf.saved_model.load(path)
  return dif_model
    

# function: to resize and save images
def save_img(img,image_size = (1084,1084),preserve_aspect_ratio = True):
  '''This function will resize and save the image'''
  global global_count
  img = tf.image.resize(img,image_size,preserve_aspect_ratio=True)
  # convert to numpy array
  img = img[0].numpy()
  # conver to range [0,255]
  img = np.uint8(img*255)
  img_info(img)
  # convert to range [0,255]
  print('Image shape',img.shape)
  # saving image
  image = Image.fromarray(np.uint8(img))
  image.save('diffused_img.png')
  

# function: to predict stylized image
def get_stylized_image(dif_model,content_img,style_img):
  content_img = tf.convert_to_tensor(content_img)
  style_img = tf.convert_to_tensor(style_img)
  output = dif_model(content_img,style_img)
  stylized_img = output[0]
  return stylized_img
  
