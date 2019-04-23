import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import imutils

# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1] 
filename = '/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/testing_data/270/frame1.jpg'
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
image_size1=list(image.shape)
#print(image_size[0])
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image1 = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image1)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/53-90-180-270-model/.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, len(os.listdir('training_data')))) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
print(result)
# for i in range(len(result)):
# 	if  result[0][0]>result[0][1]:
# 		z=270
# 		print('class 1 is 90')
# 		image = cv2.resize(image, (image_size[0], image_size[1]),0,0, cv2.INTER_LINEAR)
# 		image1=imutils.rotate_bound(image,z)
# 		cv2.imshow('san',image1)
# 		cv2.waitKey(0)

# 	else:
# 		print('class 2 is 53')
# 		c=307
# 		image = cv2.resize(image, (image_size1[0], image_size1[1]),0,0, cv2.INTER_LINEAR)
# 		image1=imutils.rotate_bound(image,c)
# 		cv2.imshow('san',image1)
# 		cv2.imwrite('/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/testing_data/53/frame1_result.jpg',image1)
# 		cv2.waitKey(0)

x= np.argmax(result)
print(x)

if x==1:
	z=90
	print('class 1 is 270')
	image = cv2.resize(image, (image_size1[0], image_size1[1]),0,0, cv2.INTER_LINEAR)
	image1=imutils.rotate_bound(image,z)
	cv2.imshow('san',image1)
	cv2.imwrite('/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/testing_data/270/frame1_result.jpg',image1)
	cv2.waitKey(0)
elif x==2:
	z=307
	print('class 1 is 53')
	image = cv2.resize(image, (image_size1[0], image_size1[1]),0,0, cv2.INTER_LINEAR)
	image1=imutils.rotate_bound(image,z)
	cv2.imshow('san',image1)
	cv2.imwrite('/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/testing_data/53/frame1_result.jpg',image1)
	cv2.waitKey(0)
elif x==0:
	z=270
	print('class 1 is 90')
	image = cv2.resize(image, (image_size1[0], image_size1[1]),0,0, cv2.INTER_LINEAR)
	image1=imutils.rotate_bound(image,z)
	cv2.imshow('san',image1)
	cv2.imwrite('/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/testing_data/90/frame1_result.jpg',image1)
	cv2.waitKey(0)
elif x==3:
	z=180
	print('class 1 is 180')
	image = cv2.resize(image, (image_size1[0], image_size1[1]),0,0, cv2.INTER_LINEAR)
	image1=imutils.rotate_bound(image,z)
	cv2.imshow('san',image1)
	cv2.imwrite('/home/kantareddy/Desktop/rotation/tutorial-2-image-classifier/testing_data/180/frame1_result.jpg',image1)
	cv2.waitKey(0)