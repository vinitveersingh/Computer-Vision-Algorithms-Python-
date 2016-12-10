import cv2
import numpy as np

def Reduce(img):
    height,width,channel = img.shape
    op_height = int(height/2.0)
    op_width  = int(width/2.0)
    output_image = np.zeros((op_height,op_width,channel),np.uint8)

    for z in range(channel):
        for y in range(op_height):
            for x in range(op_width):
                p = 0
                for j in range(-1, 1 + 1):
                    for i in range(-1, 1 + 1):
                        m = 2 * x + i
                        n = 2 * y + j
                        if m > 0 and n >0:
                           p += img[m , n, z]

                p /= 9.0
                output_image[x, y, z] = p

    return output_image

def GaussianPyramid(img1,n):
# generate Gaussian pyramid for image
    G = img1.copy()
    gaussianpyramids_img1 = [img1]
    for i in range(n):   
        G = Reduce(G)
        gaussianpyramids_img1.append(G)


    for i in range(len(gaussianpyramids_img1)):
        windowname = 'Gaussian level' + str(i) 
        cv2.imshow(windowname,gaussianpyramids_img1[i]) #image is identified by its window name


img_path = input('Enter Image Path : ')
#img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
image = cv2.imread(img_path)
n = input('Enter number of levels you want: ')
n = int(n)
GaussianPyramid(image,n)

