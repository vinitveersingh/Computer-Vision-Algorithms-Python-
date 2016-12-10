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
    return output_image

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

    return gaussianpyramids_img1
    

def GetLaplacianPyramids(gaussianpyramids_img1):
    laplacianpyramids_img1 = []
    for j in range(len(gaussianpyramids_img1)):
        if j+1 < len(gaussianpyramids_img1):
            laplacianpyramids_elements= cv2.subtract(gaussianpyramids_img1[j], cv2.pyrUp(gaussianpyramids_img1[j+1]))
            #laplacianpyramids_elements= gaussianpyramids_img1[j] - expand(gaussianpyramids_img1[j+1])
            laplacianpyramids_img1.append(laplacianpyramids_elements)
        
        else:
            laplacianpyramids_element_final= gaussianpyramids_img1[j]
            laplacianpyramids_img1.append(laplacianpyramids_element_final)

    for j in range(len(laplacianpyramids_img1)-1):
        windowname = 'Laplacian level' + str(j) 
        cv2.imshow(windowname,laplacianpyramids_img1[j]) #image is identified by its window name

def LaplacianPyramids(I,n):
    g_pry = GaussianPyramid(I,n)
    GetLaplacianPyramids(g_pry)

    

img_path = input('Enter Image Path : ')
#img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
image = cv2.imread(img_path)
n = input('Enter number of levels you want: ')
n = int(n)
LaplacianPyramids(image,n)


