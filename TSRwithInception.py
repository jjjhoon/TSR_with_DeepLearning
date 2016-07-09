#Module Author : Jonghoon Kang
#Created       : 27th June, 2016
#Modified      : 
#Module Name   : TSRwithInception.py
#Description   : TSR(Traffic Sign Recognition) application implementation via 
#                Deep Neural Network. This source code include, training and testing
#                scheme. Refer to comment blocks

import Inception_ResNet
import numpy as np

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def inputInitialize():
	testImage = tf.placeholder("float", [None, 300, 300, 3])
	testLabel = tf.placeholder("float", [None, 43])
	trainImage = 
	trainLabel = 

# Variables and Weights Initialization
# Refer to the documents "TSR_application_Jonghoon_vX.docx"
# in file name, vX means version of documents. 
# You can find document from my server directory. lol find it  

if TRAINING:
	# Weight format description
	# Refer to "TensorFlow API manual"
	# [x, y, in-dims, out-dims]
	# Example [3, 3, 32, 48]
	# 32 of 3x3 images input and generate 48 output

	# Inception-ResNet-v2 Weights initialization
	# Inception-ResNet-A weights initialization
	incResA_w1 = init_weights([1, 1, 384, 32])	
	incResA_w2 = init_weights([1, 1, 384, 32]) 
	incResA_w3 = init_weights([1, 1, 384, 32])
	incResA_w4 = init_weights([3, 3, 32, 32]) 
	incResA_w5 = init_weights([3, 3, 32, 48]) 
	incResA_w6 = init_weights([3, 3, 48, 64]) 
	incResA_w7 = init_weights([1, 1, 128, 384]) 
	
	# Inception-ResNet-B weights initialization
	incResB_w1 = init_weights([1, 1, 1152, 192]) 
	incResB_w2 = init_weights([1, 1, 1152, 128])
	incResB_w3 = init_weights([1, 7, 128, 160])
	incResB_w4 = init_weights([7, 1, 160, 192])
	incResB_w5 = init_weights([1, 1, 384, 1152])

	# Inception-ResNet-C weights initialization
	incResC_w1 = init_weights([1, 1, 2048, 192])
	incResC_w2 = init_weights([1, 1, 2048, 192])
	incResC_w3 = init_weights([1, 3, 192, 224])
	incResC_w4 = init_weights([3, 1, 224, 256])
	incResC_w5 = init_weights([1, 1, 448, 2048])

	# Inception-ResNet-Reduction-A weights initialization
	redA_w1 = init_weights([3, 3, 384, 384])
	redA_w2 = init_weights([1, 1, 384, 256])
	redA_w3 = init_weights([3, 3, 256, 256])
	redA_w4 = init_weights([3, 3, 256, 384])

	# Inception-ResNet-Reduction-B weights initialization
	redB_w1 = init_weights([1, 1, 1152, 256])
	redB_w2 = init_weights([1, 1, 1152, 256])
	redB_w3 = init_weights([1, 1, 1152, 256])
	redB_w4 = init_weights([3, 3, 256, 288])
	redB_w5 = init_weights([3, 3, 256, 288])
	redB_w6 = init_weights([3, 3, 256, 288])
	redB_w7 = init_weights([3, 3, 288, 320])

	# Inception-ResNet-Reduction-Stem weights initialization
	stem_w1 = init_weights([3, 3, 3, 32])
	stem_w2 = init_weights([3, 3, 32, 32])
	stem_w3 = init_weights([3, 3, 32, 64])
	stem_w4 = init_weights([3, 3, 64, 96])
	stem_w5 = init_weights([1, 1, 160, 64])
	stem_w6 = init_weights([1, 1, 160, 64])
	stem_w7 = init_weights([3, 3, 64, 96])
	stem_w8 = init_weights([7, 1, 64, 64])
	stem_w9 = init_weights([1, 7, 64, 64])
	stem_w10 = init_weights([3, 3, 64, 96])
	stem_w11 = init_weights([3, 3, 192, 192])

	# Full connected layer weights initialization
	fc_w1 = init_weights([1*1*1792, 1792])
	fc_w2 = init_weights([1792, 43])

else if PRE_TRAINED:
else:
	print 'In jhIception_top.py, wrong definition of PRE_TRAINED or TRAINING parameter'

if TRAINING:	# Training and Graph shows
	
	# Data Flow of Inception-ResNet Merged Architecture
	# Stem -> Inception-ResNet-A -> Reduction A -> Inception-ResNet-B -> Reduction B
	# -> Inception-ResNet-C -> Average Pooling -> Full connected Layer -> Softmax

	with tf.name_scope("Stem Architecture") as scope:
		stemOut = inception_ResNet.inceptionStemArch(	
										initImage = inImage,
										l1_w = stem_w1,
										l2_w = stem_w2,
										l3_w = stem_w3,
										l4_w = stem_w4,
										l5_1_w = stem_w5,
										l5_2_w = stem_w6,
										l6_1_w = stem_w7,
										l6_2_w = stem_w8,
										l7_w = stem_w9,
										l8_w = stem_w10,
										l9_w = stem_w11
								   )
	
	with tf.name_scope("Inception-ResNet-A") as scope:
		aOut = inception_ResNet.incResA(
					   inData = stemOut,
		               l1_1_w = incResA_w1,
		               l1_2_w = incResA_w2,
		               l1_3_w = incResA_w3,
		               l2_1_w = incResA_w4,
		               l2_2_w = incResA_w5,
		               l3_w = incResA_w6,
		               l4_w = incResA_w7
		              )
	
	with tf.name_scope("Inception-ResNet-Reduction-A") as scope:
		red_A_Out = inception_ResNet.reDuctionLayerA(
									inData = aOut,
									l1_1_w = redA_w1,
									l1_2_w = redA_w2,
									l2_w = redA_w3,
									l3_w = redA_w4
								   )
	
	with tf.name_scope("Inception-ResNet-B") as scope:
		bOut = inception_ResNet.incResB(
					   inData = red_A_Out,
					   l1_1_w = incResB_w1,
					   l1_2_w = incResB_w2,
					   l2_w = incResB_w3,
					   l3_w = incResB_w4,
					   l4_w = incResB_w5
					  )
	
	with tf.name_scope("Inception-ResNet-Reduction-B") as scope:
		red_B_Out = inception_ResNet.reDuctionLayerB(
									inData = bOut,
									l1_1_w = redB_w1,
									l1_2_w = redB_w2,
									l1_3_w = redB_w3,
									l2_1_w = redB_w4,
									l2_2_w = redB_w5,
									l2_3_w = redB_w6,
									l3_w = redB_w7
								   )
	
	with tf.name_scope("Inception-ResNet-C") as scope:
		cOut = inception_ResNet.incResC(
						inData = red_B_Out
						l1_1_w = incResC_w1,
						l1_2_w = incResC_w2,
						l2_w = incResC_w3,
						l3_w = incResC_w4,
						l4_w = incResC_w5
					  )
	
	with tf.name_scope("AveragePooling") as scope:
		# Average pooling layer and make it 1x1 full connected layer
		avgPool = tf.nn.avg_pool(cOut, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')

	
	with tf.name_scope("Full Connected Layer") as scope:
		# Insert drop out, and full connected layer + SoftMax Classifier
		fc1 = reshape(avgPool, [-1, fc1_w1.get_shape().as_list()[0]])
		fc1 = tf.nn.dropout(fc1, p_keep_fc)
		fc2 = tf.matmul(fc1, fc_w2)
		fc2 = tf.nn.dropout(fc2, p_keep_fc)
		softMax = tf.nn.softmax_cross_entropy_with_logits(fc2, label)

	result = softMax

else if PRE_TRAINED:	# Only for Inference

else:
	print 'In jhIception_top.py, wrong definition of PRE_TRAINED or TRAINING parameter'
	result = 0.0

if TRAINING:
	train_operation = tf.train.RMSPropOptimizer(learning_rate = 0.045, decay = 0,94)
	predict_op = tf.argmax(result, 1)
else if PRE_TRAINED:
	cost = tf.reduce_mean(result)
	predict_op = tf.argmax(result, 1)
else:
	print 'In jhIception_top.py, wrong definition of PRE_TRAINED or TRAINING parameter'

# Session Run Part
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for i in range():
		training_batch = zip(range(0, len(trainImage), batch_size), range(batch_size, len(trainImage), batch_size))
		
		for start, end in training_batch:
			sess.run(train_op, feed_dict={X: trainImage[start:end], Y: trainLabel[start:end], p_keep_fc: 0.8})

		test_indices = np.arange(len(testImage)) #Get a test batch
		np.random.shuffle(test_indices)
		test_indices = test_indices[0:test_size]

		print(i, np.mean(np.argmax(testLabel[test_indices], axis=1) == 
					sess.run(predict_op, feed)dict={X: testImage[test_indices], Y:testLabel[test_indices],
					p_keep_fc: 1.0}))





