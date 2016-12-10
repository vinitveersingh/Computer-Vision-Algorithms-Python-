import cv2
import numpy as np
     


def Expand(img):
    kernel = (1/16)* np.matrix('1 2 1 ; 2 4 2 ; 1 2 1').transpose()
    height,width,channel = img.shape
    op_height = height*2.0
    op_width  = width*2.0
    output_image = np.zeros((int(op_height),int(op_width),channel),np.uint8)


    for z in range(3):
        for y in range(int(op_height)):
            for x in range(int(op_width)):
                p = 0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        m = int((x - i - 1) / 2)
                        n = int((y - j - 1) / 2)
                        if m >0 and n >0:
                           p += img[m , n, z] * kernel [i+1, j+1]
                output_image[x, y, z] = p
    cv2.imshow('Scaled Image',output_image)

img_path = input('Enter Image Path : ')
#img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
image = cv2.imread(img_path)
Expand(image)
