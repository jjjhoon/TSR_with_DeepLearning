import matplotlib.pyplot as plt
import PIL
import csv
import tensorflow as tf
import random

TRAIN_SET_DIR = '/home/jonghoon/GTSRB/TrainImages'
TEST_SET_DIR = '/home/jonghoon/GTSRB/TestImages'

class ReadFiles(object):
    def __init__(self, batch_size, TRAIN_SET_DIR, TEST_SET_DIR, DEBUG=True):
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
        self._batchImages = []
        self._batchLabels = []
        print 'ReadFiles begins'

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

    def readTrainTrafficSigns(self):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    
        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
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
            gtFile.close()
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

    def ndarray2tensor(self, train_test):
        newImages = []
        newLabels = [] 
        if train_test == 'TRAIN':
            self._batchImages = self._trainImages
            self._batchLabels = self._trainLabels
        elif train_test == 'TEST':
            self._batchImages = self._testImages
            self._batchLabels = self._testLabels
        else:
            print 'In class ''ReadFiles'' ReadFiles.ndarray2tensor method argument must be ''TEST'' or ''TRAIN'''

        for i in range(0, len(images)):
            intImage = tf.convert_to_tensor(images[i], dtype=tf.float32)
            newLabels.append(tf.convert_to_tensor(images[i], dtype=tf.float32))
            newImages.append(tf.image.resize_images(intImage, 300, 300))
            if (self._DEBUG == True) & (i % 1000 == 0):
                print str(i) + 'st converting finished'
                print 'Resizing finished'

        if train_test == 'TRAIN':
            self._trainImages = newImages
            self._trainLabels = newLabels
        elif train_test == 'TEST':
            self._testImages = newImages
            self._testImages = newLabels
        else:
            print 'In class ''ReadFiles'' ReadFiles.ndarray2tensor method argument must be ''TEST'' or ''TRAIN'''
        
    def shuffleLists(self, train_test):
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
        self.ndarray2tensor(train_test)
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
        self.ndarray2tensor(train_test)
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

    def feedTrainSet(self):
        feedImages = []
        feedLabels = []
        train_test = 'TRAIN'
        self.readTrainTrafficSigns()
        self.shuffleLists(train_test)
        self.setTrainSize()
        if self._loadTrainDone:
            print 'File Queue is Ready to feed'
        else :
            print 'Failed to queueing the file queue'

        for i in range(self._trainCount, self._trainCount + batch_size):
            feedImages.append(self._trainImages)

    def feedTestSet(self):
        feedImages = []
        feedLabels = []
        train_test = 'TEST'
        self.readTestTrafficSigns()
        self.shuffleLists(train_test)
        self.setTestSize()
        if self._loadTestDone:
            print 'File Queue is Ready to feed'
        else:
            print 'Failed to queueing the file queue'

    def debugginMethod(self):
        trainImgLen = len(self._trainImages)
        trainLabLen = len(self._trainLabels)
        testImgLen = len(self._testImages)
        testLabLen = len(self._testLabels)
        print 'Number of train images are : ' + str(trainImgLen)
        print 'Number of train labels are : ' + str(trainLabLen)
        print 'Number of test images are : ' + str(testImgLen)
        print 'Number of test labels are : ' + str(testLabLen)

test = ReadFiles(batch_size = 20, TRAIN_SET_DIR = TRAIN_SET_DIR, TEST_SET_DIR = TEST_SET_DIR)
#test2 = test.loadAllImagesAndLabels()
test2 = test.readTrainTrafficSigns()
test3 = test.readTestTrafficSigns()