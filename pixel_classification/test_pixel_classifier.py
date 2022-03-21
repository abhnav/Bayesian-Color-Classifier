'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier
import pdb

if __name__ == '__main__':
  # test the classifier
  
  folder = 'data/validation/blue'
  
  pdb.set_trace()
  X = read_pixels(folder)
  pdb.set_trace()
  myPixelClassifier = PixelClassifier()
  y = myPixelClassifier.classify(X)
  
  print('Precision: %f' % (sum(y==3)/y.shape[0]))

  
