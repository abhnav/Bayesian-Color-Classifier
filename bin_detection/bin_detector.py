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
from skimage.morphology import (erosion, dilation, opening, closing)
from skimage.morphology import disk
from skimage import measure

sys.path.append('../pixel_classification')
from pixel_classifier import PixelClassifier

matplotlib.use('Qt5Agg')

class BinDetector():
  def __init__(self):
    '''
      Initilize your bin detector with the attributes you need,
      e.g., parameters of your classifier
    '''
    X, y = self.generate_data(folder='data/training')
    X = X/255.0
    # self.myPixelClassifier = PixelClassifier(X=X, y=y)

    # Use the default classifier to see how well it does
    self.myPixelClassifier = PixelClassifier()


  def add_data_for_class(self, cls, path, X, y):
      img = cv2.imread(path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      fig, ax = plt.subplots()
      fig.suptitle(f'Choose data for {cls}')
      ax.imshow(img)
      my_roi = RoiPoly(fig=fig, ax=ax, color='r')
      mask = my_roi.get_mask(img)
      X.append(img[mask,:])
      y.append(np.full(mask.sum(), cls))
      print(f"Added {y[-1].shape[0]} points")
      return

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
      if(i == 60):
        break
      print(f"file name: {filename}")
      path = os.path.join(folder,filename)
      for cls in [0,1,2,3]:
        self.add_data_for_class(cls, path, X, y)
      i = i+1
      pdb.set_trace()
      
    X = np.concatenate(X,axis=0)
    y = np.concatenate(y)
    with open("train_data.npy", 'wb') as f:
      np.save(f,X)
      np.save(f,y)
    return X,y



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
    x_test = img.reshape((-1,3)) / 255.0
    p_test = self.myPixelClassifier.classify(x_test)
    mask_img = p_test.reshape((img.shape[0], img.shape[1]))

    # If using rgb = 123 coding used in the pixel dataset
    mask_img[mask_img != 3] = 0
    mask_img[mask_img == 3] = 1

    # If using blue as any class >= 1
    # mask_img[mask_img >= 1] = 1


    # mk = mask_img[:,:,np.newaxis]
    # fig,ax = plt.subplots(2,1)
    # ax[0].imshow(img)
    # ax[1].imshow(mask_img)
    # fig.show()
    # plt.pause(2)
    # pdb.set_trace()
    
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
    img = img.astype(np.uint8)
    imgs = [img]
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    #Blur
    img = cv2.GaussianBlur(img, (7,7), 0)
    imgs.append(img)

    # dilate
    size = (3,17)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.dilate(img, kernel)
    imgs.append(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros(img.shape)
    for cnt in contours:
      # fill the contours so erosion doesn't eat from inside
      ar = cv2.contourArea(cnt)
      if(ar > 2500):
        cv2.drawContours(img, [cnt], -1, (255, 255, 255), cv2.FILLED)
    imgs.append(img)

    # erode along y axis to separate close bins, changes the bin dimensions
    size = (1, 64)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.erode(img, kernel)
    imgs.append(img)
    size = (1, 10)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.dilate(img, kernel)
    imgs.append(img)

    # erode along x axis to separate close bins, changes the bin dimensions
    size = (10, 1)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.erode(img, kernel)
    imgs.append(img)
    size = (14, 1)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.dilate(img, kernel)
    imgs.append(img)

    img = img.astype(np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # get the bounding box of contour, if not possible to be trashcan, move on
    img = np.zeros(img.shape)
    boxes = []
    for cnt in contours:
      ar = cv2.contourArea(cnt)
      if(ar > 2500):
        print("Checking contour with area: {ar}")
        # pdb.set_trace()
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w = min(rect[1][0], rect[1][1])
        h = max(rect[1][1], rect[1][0])
        print(f"Rectangle area is {w*h}, width:{w}, height:{h}, rotation:{rect[2]}")
        print(f"ratio of {h/w}")
        if(h < 1.0*w or h > 2.27*w):
          print("Strangeratio")
          continue

        est_box = cv2.boundingRect(box)
        if(est_box[2]>1.2*est_box[3]):
          print("Hey")
          continue
        boxes.append((est_box[0], est_box[1], est_box[0]+est_box[2], est_box[1]+est_box[3]))
        print("Box points are:")
        print(box)
        cv2.drawContours(img,[cnt],-1,(255,255,255),2)
        cv2.drawContours(img,[box],-1,(255,255,255),2)
        cv2.rectangle(img,(est_box[0], est_box[1]),(est_box[0]+est_box[2], est_box[1]+est_box[3]),(255,255,255),3)
    imgs.append(img)

    return boxes

    # fig, ax = plt.subplots(5,2,figsize=(10,20))
    # ax = ax.reshape(-1)
    # for i, im in enumerate(imgs):
      # ax[i].imshow(im)
    # plt.show()
    # pdb.set_trace()


        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # epsilon = 0.05*cv2.arcLength(ct,True)
        # approx = cv2.approxPolyDP(ct,epsilon,True)

    # # edges = cv2.Canny(image=ii, threshold1=0, threshold2=1) # Canny Edge
    
    # img5 = np.zeros(img.shape)
    # cv2.drawContours(img5, contours, -1, (255,255,255), 2)


    # YOUR CODE BEFORE THIS LINE
    ################################################################
    


