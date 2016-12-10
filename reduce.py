import cv2
import numpy as np
import math 
center =[]

def Convolve_1D_y(img1,kernel):
    s = kernel.shape[0]

    if s%2 != 0:   
        center.append(math.floor(s/2)+1)
        p_left = center[0] - 1
        p_right = center [0] - 1
    else:
        center.append(s/2)
        p_left = center[0] - 1
        p_right = center[0]

    zero_pad= cv2.copyMakeBorder(img1,0,0,p_left,p_right,cv2.BORDER_CONSTANT,value=[0,0,0])#top bottom left right
    o_h,o_w,c = img1.shape
    height,width,channel = zero_pad.shape
    output_image = np.zeros((o_h,o_w,c),np.uint8)

    for z in range(channel):
        for x in range(o_w):
            for y in range (o_h):
                p = 0
                for j in range(0,5):
                    m = j + y
                    if (m >=0 and m<=width):
                        p = p + (zero_pad [x , m , z] * kernel.item(j))
                output_image[x, y, z] = p
                
    return output_image

     
def Convolve_1D_x(img1,kernel):
    global center
    r = kernel.shape[1]
    p_up,p_down = None,None

    if r%2 != 0:
        center.append(int(r/2)+1)
        p_up   = center [0] - 1
        p_down = center [0] - 1
    else:
        center.append(r/2)
        p_up   = center [0] - 1
        down = center [0] 

    zero_pad= cv2.copyMakeBorder(img1,p_up,p_down,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])#top bottom left right
    o_h,o_w,c = img1.shape
    height,width,channel = zero_pad.shape
    output_image = np.zeros((o_h,o_w,c),np.uint8)
    for z in range(channel):
        for y in range (o_h):
            for x in range(o_w):
                p = 0
                for i in range(0,kernel.size):
                    n = i + x
                    if (n >=0 and n<=width):
                        p = p + (zero_pad [n ,y, z] * kernel.item( i ))
                output_image[x, y, z] = p

    return output_image                      

def Reduce(img):
    kernel = (1/16)* np.matrix('1 2 1 ; 2 4 2 ; 1 2 1').transpose()
    height,width,channel = img.shape
    op_height = int(height/2.0)
    op_width  = int(width/2.0)
    output_image = np.zeros((op_height,op_width,channel),np.uint8)
    for z in range(3):
        for y in range(int(op_height)):
            for x in range(int(op_width)):
                p = 0
                for j in range(-1, 1 + 1):
                    for i in range(-1, 1 + 1):
                        m = 2 * x + i
                        n = 2 * y + j
                        if m > 0 and n >0:
                            p += img[m , n, z] * kernel [i+1, j+1]
                output_image[x, y, z] = p
    return output_image



img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
image = cv2.imread(img_path)
kernel = (1/(10.7*2))*np.matrix([1.3,3.2,3.8,3.2,1.3])
kernel2 = kernel.transpose()
x_op = Convolve_1D_x(image,kernel)
y_op = Convolve_1D_y(x_op,kernel2)
op = Reduce(y_op)
cv2.imshow('Reduced Image',op)
