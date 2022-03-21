'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
from generate_rgb_data import read_pixels
import pdb

class PixelClassifier():
  def __init__(self, folder='../pixel_classification/data/training', X=None, y=None):
    '''
	    Initilize your classifier with any parameters and attributes you need
      Use folder to get data when X,y are not given.
      Else use the given dataset X,y
    '''
    self.classes = None
    self.ll_mean = None
    self.ll_std = None
    self.prior = None

    if X is None:
      X,y = self.read_data(folder)

    self.train(X,y)

  def read_data(self,folder):
    X1 = read_pixels(folder+'/red')
    X2 = read_pixels(folder+'/green')
    X3 = read_pixels(folder+'/blue')
    y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
    X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))
    return X,y

  def train(self,X,y):
    self.classes = np.unique(y)
    N = len(y) # N samples
    k = len(self.classes) # k classes
    dim = X.shape[1]

    self.ll_mean = np.zeros((k,dim))
    self.ll_std = np.zeros((k,dim))
    self.prior = np.zeros(k)

    for cls in np.arange(k):
      self.prior[cls] = np.sum(y==self.classes[cls]) / len(y)
      for d in np.arange(dim):
        self.ll_mean[cls,d] = X[y==self.classes[cls], d].mean()
        self.ll_std[cls,d] = X[y==self.classes[cls], d].std()
    # pdb.set_trace()
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach 
    # y = 1 + np.random.randint(3, size=X.shape[0])
    
    pr = ((X[:,np.newaxis, ...] - self.ll_mean[np.newaxis, ...])/
        (self.ll_std[np.newaxis, ...])) ** 2
    pr = np.exp(-0.5*pr) / self.ll_std[np.newaxis,...]
    pr = np.sum(np.log(pr), axis=2)
    # original prior array for train_data1:
    # ([0.88636338, 0.11363662])
    # original prior array for train_data2:
    # ([0.59548022, 0.10818053, 0.0263083 , 0.27003095])

    # self.prior = np.array([0.5, 0.50,0,0])
    # self.prior = np.array([0.82, 0.18,])

    # Good one with validation
    # self.prior = np.array([0.5,0.6,0.05])

    # Good one as well
    self.prior = np.array([0.47,0.46,0.081])

    pr = pr + np.log(self.prior)[np.newaxis,...]
    cli = np.argmax(pr, axis=1)
    return self.classes[cli]

