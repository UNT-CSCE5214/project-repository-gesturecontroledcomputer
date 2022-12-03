import argparse
import cv2
from imutils import paths
import imutils
import numpy as np
import os
import random

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Place the image path")

args = vars(ap.parse_args())
root_directory = os.getcwd()


# Function for salt_and_pepper

def salt_and_pepper_noise(image,prob):
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

for imagePath in paths.list_images(args["image"]):

	# Code for generating Salt and Pepper feature extraction 

	image = cv2.imread(imagePath)
	outline_kernel = np.array([[-1,-1,-1],
                           [-1,8,-1],
                           [-1,-1,-1]])

	res3 = cv2.filter2D(image,-1,outline_kernel, borderType=cv2.BORDER_CONSTANT) 


	# Creating Floders for pre_processed data for outline feature

	image_outline_mas = "C:\\Users\\Gagan\\Desktop\\Dataset_prep\\outline\\"
	image_outline = image_outline_mas + "\\pre_processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,res3)
	os.chdir(root_directory)


	outline_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(T, threshInv) = cv2.threshold(outline_img, 120, 255, cv2.THRESH_BINARY)
	cnts = cv2.findContours(threshInv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	crop = threshInv[y:y+h,x:x+w]
	crop_resized = cv2.resize(crop, (100, 100))

	# Creating Floder for processed data for Guassion blur
	image_outline = image_outline_mas + "\\processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,crop_resized)
	os.chdir(root_directory)

	#------------------- End of outline code ---------------------------------------------------------


	#-------------------- Guassion Blur----------------------------------------------------------------

	gray_scale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	guassian_img = cv2.GaussianBlur(gray_scale_img, (7, 7), 0)


	# Creating Floders for pre_processed data for Guassion Blur

	image_outline_mas = "C:\\Users\\Gagan\\Desktop\\Dataset_prep\\Guassion_blur\\"
	image_outline = image_outline_mas + "\\pre_processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,guassian_img)
	os.chdir(root_directory)

	
	(T, threshInv) = cv2.threshold(guassian_img, 120, 255, cv2.THRESH_BINARY)
	cnts = cv2.findContours(threshInv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	crop = threshInv[y:y+h,x:x+w]
	resized_crop_img = cv2.resize(crop, (100, 100))

	# Creating Floder for processed data for Guassion blur
	image_outline = image_outline_mas + "\\processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,resized_crop_img)
	os.chdir(root_directory)

	#------------------- End of Guassion blur ---------------------------------------------------------



	#--------------------- Applying Salt and Pepper noise -------------------------------------------------------------------------

	Salt_peper_noise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	Salt_peper_noise_img=salt_and_pepper_noise(Salt_peper_noise,0.05)


	# Creating Floders for pre_processed data for HOI

	image_outline_mas = "C:\\Users\\Gagan\\Desktop\\Dataset_prep\\salt_pepper\\"
	image_outline = image_outline_mas + "\\pre_processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,Salt_peper_noise_img)
	os.chdir(root_directory)

	
	(T, threshInv) = cv2.threshold(Salt_peper_noise_img, 120, 255, cv2.THRESH_BINARY)
	cnts = cv2.findContours(threshInv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	crop = threshInv[y:y+h,x:x+w]
	resized_crop_img = cv2.resize(crop, (100, 100))

	# Creating Floder for processed data for HOI 
	image_outline = image_outline_mas + "\\processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,resized_crop_img)
	os.chdir(root_directory)

	#--------------------------END of salt_pepper --------------------------------------------------------------
	

	# ---------------------- sobel -----------------------------------------------------------------

	
	# kernel 1 for to sharpening images
	sharpen_kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])

	# Kernel 2 for blur

	bluring_kernel = np.array([[0.0625,0.125,0.0625],
                           [0.125,0.25,0.125],
                           [0.0625,0.125,0.0625]])

	# Kernel 3 for outline

	outline_kernel = np.array([[-1,-1,-1],
                           [-1,8,-1],
                           [-1,-1,-1]])

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#applying sharpening kernel
	res1 = cv2.filter2D(image,-1,sharpen_kernel, borderType=cv2.BORDER_CONSTANT) 

	#applying bluring kernel
	res2 = cv2.filter2D(image,-1,bluring_kernel, borderType=cv2.BORDER_CONSTANT) 

	#applying the outline kernel
	res12 = cv2.filter2D(image,-1,outline_kernel, borderType=cv2.BORDER_CONSTANT)  

	res = res1 + res2 + res12


	# Creating Floders for pre_processed data for Sobel Filter

	image_outline_mas = "C:\\Users\\Gagan\\Desktop\\Dataset_prep\\sobel\\"
	image_outline = image_outline_mas + "\\pre_processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,res)
	os.chdir(root_directory)

	#image12 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	(T, threshInv) = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
	cnts = cv2.findContours(threshInv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	crop = threshInv[y:y+h,x:x+w]
	resized_crop_img = cv2.resize(crop, (100, 100))

	# Creating Floder for processed data for Sobel Filter

	image_outline = image_outline_mas + "\\processed"
	image_path1 = str(imagePath)
	image_path1 = image_path1.split("\\")
	floder_path1 = image_outline +"\\" + image_path1[-2]
	image_path1 = image_path1[-2]+"\\"+image_path1[-1]
	image_outline = image_outline +"\\"+ image_path1
	check_dir = os.path.exists(floder_path1)
	if not check_dir:
  		os.makedirs(floder_path1)
	os.chdir(floder_path1)
	cv2.imwrite(image_outline,resized_crop_img)
	os.chdir(root_directory)

	#-----------------------------End of Sobel filter-----------------------------------------------------

	print (image_outline)


# In[ ]:




