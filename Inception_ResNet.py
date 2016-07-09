#Module Author : Jonghoon Kang
#Created       : 24th June, 2016
#Modified      : 27th June, 2016 (2nd)
#Module Name   : Inception_ReNet.py
#Description   : Functional definition of Inception-ResNet components from Google
#                Inception-v4, Inception-ResNet and the Impact of Residual
#                Connections on Learning - Sergey Ioffe. et al

import tensorflow as tf

# Inception-ResNet A module definition

def incResA(inData, l1_1_w, l1_2_w, l1_3_w, l2_1_w, l2_2_w, l3_w, l4_w):	#inData dimension (?, 35, 35, X)
	# 1x1 convolution for dimension reduction
	layer1_1 = tf.nn.conv2d(inData, l1_1_w, strides=[1,1,1,1], padding='SAME')
	layer1_2 = tf.nn.conv2d(inData, l1_2_w, strides=[1,1,1,1], padding='SAME')
	layer1_3 = tf.nn.conv2d(inData, l1_3_w, strides=[1,1,1,1], padding='SAME')

	# 3x3 convolution layer for two divided data
	layer2_1 = tf.nn.conv2d(layer1_2, l2_1_w, strides=[1,1,1,1], padding='SAME')
	layer2_2 = tf.nn.conv2d(layer1_3, l2_2_w, strides=[1,1,1,1], padding='SAME')
	
	# 3x3 convolution layer for one divided data
	layer3_1 = tf.nn.conv2d(layer2_2, l3_w, strides=[1,1,1,1], padding='SAME')

	# Data merging and expand by 1x1 convolution  - Output size (?, 35, 35, Y)
	layer4 = tf.concat(3, [layer1_1, layer2_1, layer3_1])	

	# Last Layer for 1x1 convolution to expand data dimension	- Output size (?, 35, 35, X)
	layer5 = tf.nn.conv2d(layer4, l4_w, strides=[1,1,1,1], padding='SAME')
	
	# Output Filter Sum layer + Residual
	outputLayer = tf.nn.relu(tf.add(inData, layer5))

	return outputLayer

def incResB(inData, l1_1_w, l1_2_w, l2_w, l3_w, l4_w):
	# 1x1 convolution for dimension reduction
    layer1_1 = tf.nn.conv2d(inData, l1_1_w, strides=[1,1,1,1], padding='SAME')
	layer1_2 = tf.nn.conv2d(inData, l1_2_w, strides=[1,1,1,1], padding='SAME')
	    
	# 1x7 Convolution Layer
	layer2 = tf.nn.conv2d(layer1_2, l2_w, strides=[1,1,1,1], padding='SAME')
	     
	# 7x1 convolution Layer
	layer3 = tf.nn.conv2d(layer2, l3_w, strides=[1,1,1,1], padding='SAME')
	  
	# Data merging and expand by 1x1 convolution - Output size (?, 17, 17, Y)
	layer4 = tf.concat(3, [layer1_1, layer3])
	layer5 = tf.nn.conv2d(layer4, l4_w, strides=[1,1,1,1], padding='SAME')
	    
	# Output Filter Sum Layer + Residual
	outputLayer = tf.nn.relu(tf.add(inData, layer5))
	     
	return outputLayer

def incResC(inData, l1_1_w, l1_2_w, l2_w, l3_w, l4_w):
	# 1x1 convolution for dimension reduction
	layer1_1 = tf.nn.conv2d(inData, l1_1_w, strides=[1,1,1,1], padding='SAME')
	layer1_2 = tf.nn.conv2d(inData, l1_2_w, strides=[1,1,1,1], padding='SAME')

	# 1x3 convolution layer
	layer2 = tf.nn.conv2d(layer1_2, l2_w, strides=[1,1,1,1], padding='SAME')

	# 3x1 convolution layer
	layer3 = tf.nn.conv2d(layer2, l3_w, strides=[1,1,1,1], padding='SAME')

	# Data merging and expand by 1x1 convolution - Output Size (?, 8, 8, Y)
	layer4 = tf.concat(3, [layer1_1, layer3])
	layer5 = tf.nn.conv2d(layer4, l4_w, strides=[1,1,1,1], padding='SAME')

	# Output Filter Sum Layer + residual
	outputLayer = tf.nn.relu(tf.add(inData, layer5))

	return outputLayer

def reDuctionLayerA(inData, l1_1_w, l1_2_w, l2_w, l3_w):
	# Max pooling for data reduction
	mpLayer = tf.nn.max_pool(inData, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

	# Convolution and data reduction
	layer1_1 = tf.nn.conv2d(inData, l1_1_w, strides=[1,2,2,1], padding='VALID')
	layer1_2 = tf.nn.conv2d(inData, l1_2_w, strides=[1,1,1,1], padding='SAME')
	layer2 = tf.nn.conv2d(layer1_2, l2_w, strides=[1,1,1,1], padding='SAME')
	layer3 = tf.nn.conv2d(layer2, l3_w, strides=[1,2,2,1], padding-'VALID')

	# Filter concatenation
	outputLayer = tf.nn.relu(tf.concat(3, [mpLayer, layer1_1, layer3]))

	return outputLayer

def reDuctionLayerB(inData, l1_1_w, l1_2_w, l1_3_w, l2_1_w, l2_2_w, l2_3_w, l3_w):
	# Max pooling for data reduction
	mpLayer = tf.nn.max_pool(inData, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

	# 1x1 convolution layers
	layer1_1 = tf.nn.conv2d(inData, l1_1_w, strides=[1,1,1,1], padding='SAME')
	layer1_2 = tf.nn.conv2d(inData, l1_2_w, strides=[1,1,1,1], padding='SAME')
	layer1_3 = tf.nn.conv2d(inData, l1_3_w, strides=[1,1,1,1], padding='SAME')
	
	# 3x3 Convolution + data reduction layer
	layer2_1 = tf.nn.conv2d(layer1_1, l2_1_w, strides=[1,2,2,1], padding='VALID')
	layer2_2 = tf.nn.conv2d(layer1_2, l2_2_w, strides=[1,2,2,1], padding='VALID')
	layer2_3 = tf.nn.conv2d(layer1_3, l2_3_w, strides=[1,1,1,1], padding='SAME')
	layer3 = tf.nn.conv2d(layer2_3, l3_w, strides=[1,2,2,1], padding='VALID')

	# Filter concatenation and make output
	outputLayer = tf.nn.relu(tf.concat(3, [mpLayer, layer2_1, layer2_2, layer_3]))

	return outputLayer

def inceptionStemArch(initImage, l1_w, l2_w, l3_w, l4_w, l5_1_w, l5_2_w, l6_1_w, l6_2_w, l7_w, l8_w, l9_w)
	# Inception-ResNet-v2 Architecture's Initial sequence
	# This function will be parameterized for many input image size variation
	# Complex CNN + Inception Idea only - Not included Residual Network Idea
	
	layer1 = tf.nn.conv2d(initImage, l1_w, strides=[1,2,2,1], padding='VALID')
	layer2 = tf.nn.conv2d(layer1, l2_w, strides=[1,1,1,1], padding='VALID')
	layer3 = tf.nn.conv2d(layer2, l3_w, strides=[1,1,1,1], padding='SAME')
	layer4_1 = tf.nn.max_pool(layer3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	layer4_2 = tf.nn.conv2d(layer3, l4_w, strides=[1,2,2,1], padding='VALID')
	layer4 = tf.concat(3, [layer4_1, layer4_2])
	layer5_1 = tf.nn.conv2d(layer4, l5_1_w, strides=[1,1,1,1], padding='SAME')
	layer5_2 = tf.nn.conv2d(layer4, l5_2_w, strides=[1,1,1,1], padding='SAME')
	layer6_1 = tf.nn.conv2d(layer5_1, l6_1_w, strides=[1,1,1,1], padding='VALID')
	layer6_2 = tf.nn.conv2d(layer5_2, l6_2_w, strides=[1,1,1,1], padding='SAME')
	layer7 = tf.nn.conv2d(layer6_2, l7_w, strides=[1,1,1,1], padding='SAME')
	layer8 = tf.nn.conv2d(layer7, l8_w, strides=[1,1,1,1], padding='VALID')
	layer9 = tf.concat(3, [layer6_2, layer8])
	layer9_1 = tf.nn.conv2d(layer9, l9_w, strides=[1,2,2,1], padding='VALID')
	layer9_2 = tf.nn.max_pool(layer9, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
	outputLayer = tf.nn.relu(tf.concat(3, [layer9_1, layer9_2]))

	return outputLayer

# End of Inception_ResNet file
