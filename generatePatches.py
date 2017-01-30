import cv2
import numpy as np
from scipy.misc import imrotate
from random import shuffle,sample

PATCH_SIZE = 32
IN = 'Set1/IN/3096.jpg'
EDGE = 'Set1/EDGE/3096.jpg'
OUT_POS = 'Set1/OUT/pos/'
OUT_NEG = 'Set1/OUT/neg/'

def getData():
	img = cv2.imread(IN,0)
	mask = cv2.imread(EDGE,0)
	h,w = img.shape[0],img.shape[1]
	img = img[100:-25,150:-25]
	mask = mask[100:-25,150:-25]
	scale = .5
	img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
	mask = cv2.resize(mask,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
	mask[mask>127]=255
	mask[mask<=127]=0
	return img,mask

def display(img):
	cv2.imshow('Window',img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def genPatch(img,mask):
	height,width = img.shape[0],img.shape[1]
	print(height,width)
	p = PATCH_SIZE//2
	posImgs = []
	negImgs = []
	for x in range(p,width-p):
		for y in range(p,height-p):
			if(mask[y,x]==0):
				patch = img[y-p:y+p,x-p:x+p]
				flipx = cv2.flip(patch,0)
				flipy = cv2.flip(patch,1)
				flipxy = cv2.flip(patch,-1)
				rot90 = imrotate(patch,90)
				rot180 = imrotate(patch,180)
				rot270 = imrotate(patch,270)
				posImgs.append(patch)
				posImgs.append(flipx)
				posImgs.append(flipy)
				posImgs.append(flipxy)
				posImgs.append(rot90)
				posImgs.append(rot180)
				posImgs.append(rot270)
				# countPos+=1
				# filename = OUT_POS+str(countPos)+'.jpg'
				# print(filename)
				# cv2.imwrite(filename,patch)
			elif(mask[y,x]==255):
				patch = img[y-p:y+p,x-p:x+p]
				negImgs.append(patch)
				# countNeg+=1
				# filename = OUT_NEG+str(countNeg)+'.jpg'
				# print(filename)
				# cv2.imwrite(filename,patch)
			else:
				print("Error at : ",y,x)
	return posImgs,negImgs

def main():
	img,mask = getData()
	out = np.hstack([img,mask])

	display(out)
	posImgs,negImgs = genPatch(img,mask)
	print(len(posImgs))
	posImgs = sample(posImgs,2000)
	negImgs = sample(negImgs,2*len(posImgs))
	shuffle(negImgs)
	print(len(negImgs))
	trainSet = posImgs+negImgs
	trainLabel = np.zeros((len(trainSet)),np.float32)
	trainLabel[:len(posImgs)]=1
	print(trainLabel)
	np.save("train",trainSet)
	np.save("label",trainLabel)
	
	img2,mask2 = cv2.flip(img,0),cv2.flip(mask,0)
	posImgs,negImgs = genPatch(img2,mask2)
	posImgs = sample(posImgs,2000)
	negImgs = sample(negImgs,2*len(posImgs))
	testSet = posImgs+negImgs
	testLabel = np.zeros((len(testSet)),np.float32)
	testLabel[:len(posImgs)]=1
	np.save("test",testSet)
	np.save("testlabel",testLabel)

	trainSet = np.load("train.npy")
	trainLabel = np.load("label.npy")
	print(trainSet.shape)
	print(trainLabel.shape)
	print(np.count_nonzero(trainLabel))

	testSet = np.load("train.npy")
	testLabel = np.load("label.npy")
	print(testSet.shape)
	print(testLabel.shape)
	print(np.count_nonzero(testLabel))

	X, Y, testX, testY = trainSet,trainLabel,testSet,testLabel
	X = X.reshape([-1, 32, 32, 1])
	testX = testX.reshape([-1, 32, 32, 1])
	print(testX.shape)


main()