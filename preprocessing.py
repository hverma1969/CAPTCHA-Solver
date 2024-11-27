import cv2
import numpy as np
# from remove_line import *
# path="naive3.jpeg"
# lower_threshold=30
# upper_threshold=150
def preprocess(img):
	kernel=np.ones((5,5))
	gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	width=gray_img.shape[1]
	height=gray_img.shape[0]
	# print("heighttttttttt",height)
	# print("widthhhhhhhhhhhhhhh",width)
	l_threshold=int(height/8)
	u_threshold=int(height*1.5)
	tl=gray_img[height-2][2]
	tr=gray_img[height-2][width-2]
	bl=gray_img[2][2]
	br=gray_img[2][width-2]
	tc=gray_img[height-2][int(width/2)-2]
	bc=gray_img[2][int(width/2)-2]
	# print(tl,tr,bl,br,tc,bc)
	count=0
	if tl<110:
		count=count+1
	if tr<110:
		count=count+1
	if bl<101:
		count=count+1
	if br<110:
		count=count+1
	if tc<110:
		count=count+1
	if bc<110:
		count=count+1
	# if(count<=3):
	# print("1111111111111111111111111111111111111",count)

	if count>5:
		gray_img=cv2.bitwise_not(gray_img)
	# gray_img = cv2.medianBlur(gray_img,5)
	# cv2.imshow("med",gray_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
	gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
	# gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
	gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
	gray_img = cv2.bitwise_not(gray_img)

	# gray_img = cv2.medianBlur(gray_img,5)
	# cv2.imshow("med",gray_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# new_img=preprocessing.pre_proc_img(gray_img)

	contours,_ = cv2.findContours(cv2.bitwise_not(gray_img.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects=[]
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		# if w<lower_threshold and h < lower_threshold:
		# 	if w>h:
		# 		r=w
		# 	else:
		# 		r=h
		# 	gray_img=cv2.circle(gray_img,(int(x+w/2),int(y+h/2)),r,(255,255,255),-1)
		# 	continue
		if w<l_threshold or h< l_threshold:
			if w>h:
				r=w
			else:
				r=h
			gray_img=cv2.circle(gray_img,(int(x+w/2),int(y+h/2)),r,(255,255,255),-1)
			# continue

		elif w>u_threshold:
			rects.append((x,y,int(w/2),h))
			rects.append((x+int(w/2),y,int(w/2),h))
			# continue
		else:
			rects.append((x,y,w,h))
		# elif w<lower_threshold and h>lower_threshold:

		# 	rects.append(())	
		# print(w)
		# imCrop = cv2.circle(imCrop, (x+w,y), 4, (0,0,255), 3)
		# cv2.rectangle(gray_img,(x,y),(x+w,y+h),(0,255,0),2)
	# cv2.imshow("med",gray_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()	
	images=[]
	images2=[]
	rects.sort()
	# cv2.imshow('d',gray_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	for rect in rects:
		x=rect[0]
		y=rect[1]
		w=rect[2]
		h=rect[3]
		# print(x,y,h,w)
		new_img=gray_img[y:y+h,x:x+w]
		new_img=cv2.resize(new_img,(100,100))
		new_img=cv2.bitwise_not(new_img)
		new_img=cv2.copyMakeBorder(new_img,40,40,40,40,cv2.BORDER_CONSTANT)
		# new_img=cv2.bitwise_not(new_img)
		new_img = cv2.dilate(new_img,np.ones((5,5)),iterations = 1)
		new_img1=cv2.resize(new_img,(64,64))
		new_img1 = cv2.bitwise_not(new_img1)
		new_img2=cv2.resize(new_img,(28,28))
		img_exp1 = np.expand_dims(new_img1, axis=2)
		img_exp2 = np.expand_dims(new_img2, axis=2)
		img_f1=np.expand_dims(img_exp1,axis=0)
		img_f2=np.expand_dims(img_exp2,axis=0)
		img_f1=img_f1/255.0
		img_f2=img_f2/255.0
		images.append(img_f1)
		images2.append(img_f2)
		# cv2.imshow("s",new_img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	# return rects	
	return images,images2
	# return images
# print(rects[0][0])
# cv2.imshow('img',imCrop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()