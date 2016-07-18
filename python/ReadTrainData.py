'''
Module Author : Jonghoon Kang
Created       : 24th June, 2016
Modified      : 27th June, 2016 (2nd)
Module Name   : ReadTrainData.py
Description   : Read images from GTSRB data set (Test and Train dataset both)
				Refer to GTSRB data set - http://benchmark.ini.rub.de/
				+
				You must download the dataset before use this code
				- In test dataset you must get "Extended annotations including class ids" csv file
				Reading DB is referenced by GTSRB organizer's engineer
Usage          : Use function for 
					-Reading training set
						ReadFiles.feedTrainSet()
					-Reading test set
						ReadFiles.feedTestSet()
					Above functions are return images and labels in amount of "Batch_size"
'''
import matplotlib.pyplot as plt
import PIL
import Image
import csv
import tensorflow as tf
import random
import os

TRAIN_SET_DIR = '/home/jonghoon/GTSRB/TrainImages'
TEST_SET_DIR = '/home/jonghoon/GTSRB/TestImages'

class ReadFiles(object):
	def __init__(self, batch_size, TRAIN_SET_DIR, TEST_SET_DIR, DEBUG=False):
		self._DEBUG = DEBUG
		self._batch_size = batch_size
		self._trainSize = 0     # Number of train images/labels 
		self._testSize = 0      # Number of test images/labels
		self._trainCount = 0    # Counter of batch and its size
		self._testCount = 0       
		self._trainImages = []
		self._trainLabels = []
		self._testImages = []
		self._testLabels = []
		self._loadTestDone = False
		self._loadTrainDone = False
		self._TRAIN_SET_DIR = TRAIN_SET_DIR
		self._TEST_SET_DIR = TEST_SET_DIR
		self._TRAIN_SET_DIR_LINEAR = TRAIN_SET_DIR + '/linearTestSet/'
		self._batchImages = []
		self._batchLabels = []
		self._trainReOrdered = False
		if DEBUG:
			print 'ReadFiles begins'

	def debuggingMethod(self):
		trainImgLen = len(self._trainImages)
		trainLabLen = len(self._trainLabels)
		testImgLen = len(self._testImages)
		testLabLen = len(self._testLabels)
		print 'Number of train images are : ' + str(trainImgLen)
		print 'Number of train labels are : ' + str(trainLabLen)
		print 'Number of test images are : ' + str(testImgLen)
		print 'Number of test labels are : ' + str(testLabLen)
	
	def debug2print(self):
		for i in range(0, len(self._batchLabels)):
			print 'BatchLabels ' + str(i)
			print '       ' + str(self._batchLabels)

	#@trainCount.setter
	#def trainCount_setter(self, num):
	#    print '_trainCount setter called'
	#    self._trainCount = num
	#
	#@testCount.setter
	#def testCount_setter(self, num):
	#    print '_testCount setter called'
	#    self._testCount = num

	@property
	def trainSize(self):
		print '_trainSize getter called'
		return self._trainSize

	@property
	def testSize(self):
		print '_testSize getter called'
		return self._testSize
	
	@property
	def testCount(self):
		print '_trainCount getter called'
		return self._testCount

	@property
	def trainCount(self):
		print '_trainCount getter called'
		return self._trainCount
	
	@property
	def batch_size(self):
		print '_batch_size getter called'
		return self._batch_size
	
	#@batchSize.setter
	#def batchSize_setter(self, num):
	#    print '_batch_size setter called'
	#    self._batch_size = num

    #@trainCount.setter
    #def trainCount_setter(self, num):
    #    print '_trainCount setter called'
    #    self._trainCount = num
    #
    #@testCount.setter
    #def testCount_setter(self, num):
    #    print '_testCount setter called'
    #    self._testCount = num

	@property
	def trainSize(self):
		print '_trainSize getter called'
		return self._trainSize

	@property
	def testSize(self):
		print '_testSize getter called'
		return self._testSize
	
	@property
	def testCount(self):
		print '_trainCount getter called'
		return self._testCount

	@property
	def trainCount(self):
		print '_trainCount getter called'
		return self._trainCount
	
	@property
	def batch_size(self):
		print '_batch_size getter called'
		return self._batch_size
    
    #@batchSize.setter
    #def batchSize_setter(self, num):
    #    print '_batch_size setter called'
    #    self._batch_size = num

	#def maybeReStoreTrainData(self):
	#	# This method is called when the initial codes called form top architecture
	#	# to store the GTSRB data set into serial directory
	#
	#	if not os.path.exists(self._TRAIN_SET_DIR_LINEAR):
	#		os.mkdir(self._TRAIN_SET_DIR_LINEAR)
	#		print 'New directory successfully created in ' + self._TRAIN_SET_DIR_LINEAR
	#
	#	#for i in range(0, len(images)):


	def readTrainTrafficSigns(self):
		images = [] # images
		labels = [] # corresponding labels
		# loop over all 42 classes
		for c in range(0,43):
			prefix = self._TRAIN_SET_DIR + '/' + format(c, '05d') + '/' # subdirectory for class
			if self._DEBUG == True:
				print prefix
			gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
			gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
			gtReader.next() # skip header
			# loop over all images in current annotations file
			for row in gtReader:
				rImage = plt.imread(prefix + row[0])
				images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
				labels.append(row[7]) # the 8th column is the label
				#if self._DEBUG:
				#	fName = prefix + row[0]
				#	print 'Read File Name : ' + fName

			gtFile.close()

			#if self._DEBUG:
			#	for x in range(0, len(images)):
			#		plt.imshow(images[x])
			#		plt.show()

		self._trainImages = images
		self._trainLabels = labels

	def readTestTrafficSigns(self):
		images = []
		labels = []
		prefix = self._TEST_SET_DIR + '/' + 'GT-final_test.csv'
		if self._DEBUG:
			print prefix
		gtFile = open(prefix)
		gtReader = csv.reader(gtFile, delimiter=';')
		gtReader.next() # skip header
		for row in gtReader:
			rImage = plt.imread(TEST_SET_DIR + '/' + row[0])
			images.append(plt.imread(TEST_SET_DIR + '/' + row[0]))
			labels.append(row[7])
		gtFile.close()
		self._testImages = images
		self._testLabels = labels

	def ndarray2tensor(self, train_test, feed = True):
		if feed == False:
			newImages = []
			#newLabels = [] 
			if train_test == 'TRAIN':
				images = self._trainImages
				labels = self._trainLabels
			elif train_test == 'TEST':
				images = self._testImages
				labels = self._testLabels
			else:
				print 'In class ''ReadFiles'' ReadFiles.ndarray2tensor method argument must be ''TEST'' or ''TRAIN'''

		if self._DEBUG:
			print 'ReadFiles.ndarray2tensor(), Start to typecasting and resizing'
		
		if feed:
			for i in range(0, len(self._batchImages)):
				intImage = tf.convert_to_tensor(self._batchImages[i], dtype=tf.float32)
				self._batchImages[i] = tf.image.resize_images(intImage, 300, 300)
				#self._batchLabels[i] = tf.convert_to_tensor(self._batchLabels[i], dtype=tf.int32)
		else:
			for i in range(0, len(images)):
				intImage = tf.convert_to_tensor(images[i], dtype=tf.float32)
				#newLabels.append(tf.convert_to_tensor(labels, dtype=tf.int32))
				newImages.append(tf.image.resize_images(intImage, 300, 300))
				if (self._DEBUG == True) & (i % 1000 == 0):
					print str(i) + 'st converting finished'
					print 'Resizing finished'

		if feed == False:
			if train_test == 'TRAIN':
				self._trainImages = newImages
				#self._trainLabels = newLabels
			elif train_test == 'TEST':
				self._testImages = newImages
				#self._testLabels = newLabels
			else:
				print 'In class ''ReadFiles'' ReadFiles.ndarray2tensor method argument must be ''TEST'' or ''TRAIN'''

		if self._DEBUG:
			print 'ReadFiles.ndarray2tensor(), finished to typecasting and resizing'

	def shuffleLists(self, train_test):
		if self._DEBUG:
			print 'ReadFiles.shuffleLists(), begin to shuffle ndarrays'

		if train_test == 'TRAIN':
			sortedImages = self._trainImages
			sortedLabels = self._trainLabels
		elif train_test == 'TEST':
			sortedImages = self._testImages
			sortedLabels = self._testLabels
		else:
			print 'In class ''ReadFiles'' ReadFiles.shuffleLists method argument must be ''TEST'' or ''TRAIN'''

		shuffleData = zip(sortedImages, sortedLabels)
		random.shuffle(shuffleData)

		if train_test == 'TRAIN':
			self._trainImages, self._trainLabels = zip(*shuffleData)
		elif train_test == 'TEST':
			self._testImages, self._testLabels = zip(*shuffleData)
		else:
			print 'In class ''ReadFiles'' ReadFiles.shuffleLists method argument must be ''TEST'' or ''TRAIN'''

		if self._DEBUG:
			print 'ReadFiles.shuffleLists(), finished to shuffle ndarrays'

	def setTrainSize(self):
		if len(self._trainImages) == len(self._trainLabels):
			self._trainSize = len(self._trainImages)
			return True
		else:
			self._trainSize = 0
			print 'Failed to load train set in class ReadFiles.setTrainSize()\nNumber of images and labels are not matched'
			return False

	def setTestSize(self):
		if len(self._testImages) == len(self._testLabels):
			self._testSize = len(self._trainImages)
			return True
		else:
			self._trainSize = 0
			print 'Failed to load train set in class ReadFiles.setTrainSize()\nNumber of images and labels are not matched'
			return False

	# Recommend you to not to use this method (loadTrainSet)
	# This method can cause your system shutdown 
	# due to memory usage (Upto 8GB + Swap 4GB)
	def loadTrainSet(self):
		train_test = 'TRAIN'
		self.readTrainTrafficSigns()
		self.ndarray2tensor(train_test, feed = False)
		self.shuffleLists(train_test)
		self._loadTrainDone = self.setTrainSize()
		if self._loadTrainDone:
			print 'Loading Training Dataset done'
		else:
			print 'Failed to load train set'

	# Recommend you to not to use this method (loadTestSet)
	# This method can cause your system shutdown 
	# due to memory usage (Upto 8GB + Swap 4GB)
	def loadTestSet(self):
		train_test = 'TEST'
		self.readTestTrafficSigns()
		self.ndarray2tensor(train_test, feed = False)
		self.shuffleLists(train_test)
		self.setTestSize()
		self._loadTestDone = self.setTestSize()
		if self._loadTestDone:
			print 'Loading Test dataset done'
		else:
			print 'Failed to load test set'

	# Recommend you to not to use this method (loadAllImagesAndLabels)
	# This method can cause your system shutdown 
	# due to memory usage (Upto 8GB + Swap 4GB)
	def loadAllImagesAndLabels(self):
		self.loadTrainSet()
		self.loadTestSet()
		if self._loadTestDone & self._loadTrainDone:
			print 'Complete to load all test and train set'
		else:
			print 'Failed to load all test and train set'

	def feedTrainSet(self, initStatus = False):
		self._batchImages = []
		self._batchLabels = []
		train_test = 'TRAIN'
		if initStatus:
			self.readTrainTrafficSigns()
			self.shuffleLists(train_test)
			self.setTrainSize()
			if self._loadTrainDone:
				print 'File Queue is Ready to feed'
			else:
				print 'Failed to quieing the file queue'

		for i in range(self._trainCount, self._trainCount + self._batch_size):
			self._batchImages.append(self._trainImages[i])
			self._batchLabels.append(self._trainLabels[i])

		if self._DEBUG:
			print 'BatchLabels ' + str(self._batchLabels)
			for i in range(0, len(self._batchImages)):
				plt.imshow(self._batchImages[i])
				plt.show()

			self._batchImages.append(self._trainImages[self._trainCount])
			self._batchLabels.append(self._trainLabels[self._trainCount])

		if self._DEBUG:
			print 'BatchLabels ' + str(self._batchLabels)
			#for i in range(0, len(self._batchImages)):
			#	plt.imshow(self._batchImages[i])
			#	plt.show()

			self._batchImages.append(self._trainImages[i])
			self._batchLabels.append(self._trainLabels[i])

		if self._DEBUG:
			print 'BatchLabels ' + str(self._batchLabels)
			for i in range(0, len(self._batchImages)):
				plt.imshow(self._batchImages[i])
				plt.show()

		self._trainCount += self._batch_size
		self.ndarray2tensor(train_test)

		if self._DEBUG:
			print 'Value of ''_trainCount'' : ' + str(self._trainCount)
			print 'Size of ''_batchImages'' : ' + str(len(self._batchImages))
			print 'Size of ''_batchLabels'' : ' + str(len(self._batchLabels))

	def feedTestSet(self, initStatus = False):
		feedImages = []
		feedLabels = []
		train_test = 'TEST'
		if initStatus:
			self.readTestTrafficSigns()
			self.shuffleLists(train_test)
			self.setTestSize()
			if self._loadTestDone:
				print 'File Queue is Ready to feed'
			else:
				print 'Failed to queueing the file queue'

		for i in range(self._testCount, self._testCount + self._batch_size):
			self._batchImages.append(self._testImages[self._testCount])
			self._batchLabels.append(self._testLabels[self._testCount])

		if self._DEBUG:
			print 'BatchLabels ' + str(self._batchLabels)

		self._testCount += self._batch_size
		self.ndarray2tensor(train_test)
		if self._DEBUG:
			print 'Value of ''_testCount''  : ' + str(self._testCount)
			print 'Size of ''_batchImages'' : ' + str(len(self._batchImages))
			print 'Size of ''_batchLabels'' : ' + str(len(self._batchLabels))

test = ReadFiles(batch_size = 20, TRAIN_SET_DIR = TRAIN_SET_DIR, TEST_SET_DIR = TEST_SET_DIR, DEBUG = True)
#test2 = test.loadAllImagesAndLabels()
#test2 = test.readTrainTrafficSigns()
#test3 = test.readTestTrafficSigns()
flag = True
for i in range(0, 1000):
	test4 = test.feedTrainSet(initStatus = flag)
	flag = False

test.debuggingMethod()