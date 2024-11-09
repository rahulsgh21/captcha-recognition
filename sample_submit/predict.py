# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR
import os
import glob
import cv2 as cv
import numpy as np
from PIL import Image
import os
from os import listdir
import imutils
from matplotlib import pyplot as plt
# INPUT CONVENTION
# filenames: a list of strings containing filenames of images
def rotatea(template,angle):
    rows = template.shape[0]
    cols = template.shape[1]

    img_center = (cols / 2, rows / 2)
    M = cv.getRotationMatrix2D(img_center, angle, 1)

    template = cv.warpAffine(template, M, (cols, rows), borderValue=(255,255,255))

    return template

def cleanim(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (0,0), sigmaX=33, sigmaY=33)
    divide = cv.divide(img, blur, scale=255)
    thresh = cv.threshold(divide, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    return morph
def remove_adjacent(nums):
  i = 1
  while i < len(nums):
    if nums[i] == nums[i-1]:
      nums.pop(i)
      i -= 1
    i += 1
  return nums
# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

def decaptcha( filenames ):
	img_dir = "../reference" # Enter Directory of all images
	data_path = os.path.join(img_dir,'*g')
	files = glob.glob(data_path)
	labels=[]
	for imgg in filenames:
		img_rgb = cv.imread(imgg)
		img_gray=cleanim(img_rgb)
		data=[]
		for f1 in files:
			for i in range(-30,40,10):
				img = cv.imread(f1)
				img=rotatea(img,i)
				img=cleanim(img)
				w, h = img.shape[::-1]
				res = cv.matchTemplate(img_gray,img,cv.TM_CCOEFF_NORMED)
				threshold = 0.70
				loc = np.where( res >= threshold)
				if(loc[0].size>0 and loc[1].size>0):
					for pt in zip(*loc[::-1]):
						cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
						data.append([f1[13:-4],pt])
		data = sorted(data,key=lambda k:k[1])
		findata=[]
		for k in data:
			findata.append(k[0])
		findata=remove_adjacent(findata)
		findata = ','.join(findata)
		labels.append(findata)
	# The use of a model file is just for sake of illustration
	# with open( "model.txt", "r" ) as file:
	# 	labels = file.read().splitlines()
	return labels


