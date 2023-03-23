import numpy as np
import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from skimage.util import random_noise
import cv2
import time


# Converting without mask RGB to B/W and back to RGB
path = "C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\without_mask"
j = 0
from PIL import Image
for img in os.listdir(path):
    img_path = os.path.join(path, img)
    img = Image.open(img_path)
    img = img.convert("L")
    img = img.convert("RGB")
    img.save("C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\without_mask_grey\\without_mask_"+str(j)+"_grey.jpg")
    j = j+1

# Converting with mask RGB to B/W and back to RGB
for img in os.listdir(path):
    img_path = os.path.join(path, img)
    img = Image.open(img_path)
    img = img.convert("L")
    img = img.convert("RGB")
    img.save("C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\with_mask_grey\\with_mask_"+str(j)+"_grey.jpg")
    j = j+1


path = "C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\with_mask_grey"
j = 0
for img in os.listdir(path):
    # Load the image
    img_path = os.path.join(path, img)
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (5, 5), 1)
    cv2.imwrite("C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\with_mask_CCTV\\CCTV_Image_"+str(j)+'.jpg',blurred)
    j=j+1




path = "C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\without_mask_grey"
j = 0
for img in os.listdir(path):
    img_path = os.path.join(path, img)
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (5, 5), 1)
    cv2.imwrite("C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset\\without_mask_CCTV\\CCTV_Image_"+str(j)+'.jpg',blurred)
    j=j+1





