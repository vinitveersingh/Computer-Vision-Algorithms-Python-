import cv2
import numpy as np
import math
center = []

def Convolve(img1,kernel):
    global center
    r = kernel.shape[0]
    s = kernel.shape[1]
    p_up,p_down,p_left,p_right = None,None,None,None


    if r%2 != 0:
        center.append(int(r/2)+1)
        p_up   = center [0] - 1
        p_down = center [0] - 1
    else:
        center.append(r/2)
        p_up   = center [0] - 1
        p_down = center [0] 

    if s%2 != 0:   
        center.append(math.floor(s/2)+1)
        p_left = center[1] - 1
        p_right = center [1] - 1
    else:
        center.append(s/2)
        p_left = center[1] - 1
        p_right = center[1]

    
    zero_pad= cv2.copyMakeBorder(img1,p_up,p_down,p_left,p_right,cv2.BORDER_CONSTANT,value=[0,0,0])#top bottom left right  
    o_h,o_w,c = img1.shape
    height,width,channel = zero_pad.shape
    output_image = np.zeros((o_h,o_w,c),np.uint8)

    for z in range(channel):
        for y in range(o_h):
            for x in range(o_w):
                p = 0
                for j in range(0,3):
                    for i in range(0,3):
                            m = i + x
                            n = j + y
                            if (m >=0 and m<=height) and (n >=0 and n<=width):
                                p =(p + zero_pad [m ,n, z] * kernel [i, j])
                                
                output_image[x, y, z] = p
                
    cv2.imshow('Convoluted Image',output_image)


img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
image = cv2.imread(img_path)
y_n = input('Choose default kernel? Press y for yes or n for no ')
if y_n == 'y':
    kernel = (1/16)* np.matrix('1 2 1 ; 2 4 2 ; 1 2 1').transpose()
else:
    u_kernel = input('Enter kernel in numpy format: ')
    kernel = (1/16)* np.matrix(u_kernel).transpose()

Convolve(image,kernel)

