'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import sys
import numpy as np
import os, cv2
from roipoly import RoiPoly
import pdb
import matplotlib
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

sys.path.append('../pixel_classification')
from pixel_classifier import PixelClassifier

matplotlib.use('Qt5Agg')

class BinDetector():
  def __init__(self):
    '''
      Initilize your bin detector with the attributes you need,
      e.g., parameters of your classifier
    '''
    X,y = self.generate_data(folder='data/training')
    myPixelClassifier = PixelClassifier(X=X, y=y)

  def generate_data(self, folder='data/training'):
    # search for stored data first
    try:
      with open("train_data.npy", 'rb') as f:
        X = np.load(f)
        y = np.load(f)
      return X,y
    except:
      print(f"training data does not exist. Creating now.")

    X = []
    y = []
    i = 0
    for filename in os.listdir(folder):  
      if(i == 2):
        break
      img = cv2.imread(os.path.join(folder,filename))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      print(f"file name: {filename}")
      # Get blue from image
      fig, ax = plt.subplots()
      fig.suptitle('Choose trashcan blue positive')
      ax.imshow(img)
      my_roi = RoiPoly(fig=fig, ax=ax, color='r')
      mask = my_roi.get_mask(img)
      X.append(img[mask,:])
      y.append(np.ones(mask.sum()))

      fig, ax = plt.subplots()
      fig.suptitle('Choose NOT trashcan blue')
      ax.imshow(img)
      my_roi = RoiPoly(fig=fig, ax=ax, color='r')
      mask = my_roi.get_mask(img)
      X.append(img[mask,:])
      y.append(np.zeros(mask.sum()))
      i = i+1
    X = np.concatenate(X,axis=0)
    y = np.concatenate(y)
    pdb.set_trace()
    with open("train_data.npy", 'wb') as f:
      np.save(f,X)
      np.save(f,y)



  def segment_image(self, img):
    '''
      Obtain a segmented image using a color classifier,
      e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
      call other functions in this class if needed
      
      Inputs:
        img - original image
      Outputs:
        mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Replace this with your own approach 
    mask_img = img
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return mask_img

  def get_bounding_boxes(self, img):
    '''
      Find the bounding boxes of the recycling bins
      call other functions in this class if needed
      
      Inputs:
        img - original image
      Outputs:
        boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
        where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Replace this with your own approach 
    x = np.sort(np.random.randint(img.shape[0],size=2)).tolist()
    y = np.sort(np.random.randint(img.shape[1],size=2)).tolist()
    boxes = [[x[0],y[0],x[1],y[1]]]
    boxes = [[182, 101, 313, 295]]
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    
    return boxes


