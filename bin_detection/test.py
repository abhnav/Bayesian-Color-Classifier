import os, cv2
import yaml
import pdb
from bin_detector import BinDetector
from matplotlib import pyplot as plt

folder = "data/diff"
my_detector = BinDetector()
for filename in os.listdir(folder):
  if filename.endswith(".jpg"):
    # read one test image
    print(f"doing file: {filename}")
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_img = my_detector.segment_image(img)
    bb = my_detector.get_bounding_boxes(mask_img)
