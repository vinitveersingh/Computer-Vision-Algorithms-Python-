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
            laplacianpyramids_elements= gaussianpyramids_img1[j] - Expand(gaussianpyramids_img1[j+1])
            laplacianpyramids_img1.append(laplacianpyramids_elements)
        
        else:
            laplacianpyramids_element_final= gaussianpyramids_img1[j]
            laplacianpyramids_img1.append(laplacianpyramids_element_final)

    return laplacianpyramids_img1

def reconstruct_g(laplacianpyramids_element_final,laplacianpyramids_img1,gaussianpyramids_img1):
    reconstructed_gaussian_level = laplacianpyramids_element_final
    reconstructed_gaussian_img1 = [reconstructed_gaussian_level]
    reconstructed_counter = 0
    display_counter= 0
    display_counter2 = 0
    for k in reversed(range(len(laplacianpyramids_img1)-1)):
        reconstructed_gaussian_level = laplacianpyramids_img1[k] + Expand(reconstructed_gaussian_img1[reconstructed_counter])
        reconstructed_counter+=1
        reconstructed_gaussian_img1.append(reconstructed_gaussian_level)
        if reconstructed_counter == len(laplacianpyramids_img1):
            break

    for j in reversed(range(len(reconstructed_gaussian_img1))):
        window_name= 'Reconstructed Images' + str(display_counter)
        display_counter+=1
        cv2.imshow(window_name,reconstructed_gaussian_img1[j]) 


    for l in reversed(range(len(reconstructed_gaussian_img1))):
        window_name= 'Error at Gaussian Level' + str(display_counter2)
        cv2.imshow(window_name,gaussianpyramids_img1[display_counter2] - reconstructed_gaussian_img1[l]) 
        display_counter2+=1

def Reconstruct(LI,n):
    g_pyr = GaussianPyramid(LI,n)
    l_pyr = GetLaplacianPyramids(g_pyr)
    reconstruct_g(l_pyr[len(l_pyr)-1],l_pyr,g_pyr)
    
    
img_path = input('Enter Image Path : ')
#img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
image = cv2.imread(img_path)
n = input('Enter number of levels you want: ')
n = int(n)
Reconstruct(image,n)
    


