import cv2
import numpy as np
import math
import time

def create_window(window_name):
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

def destroy_window(window_name):
    cv2.destroyWindow(window_name,cv2.WINDOW_AUTOSIZE)

def show_image(window_name,img):
    cv2.imshow(window_name,img)


def pad(img1,img2,window_size):
    center = []
    r = window_size[0]
    s = window_size[1]
    p_up,p_down,p_left,p_right = None,None,None,None

    if r%2 != 0:
        center.append(int(r/2)+1)
        p_up   = int(center [0] - 1)
        p_down = int(center [0] - 1)
    else:
        center.append(r/2)
        p_up   = int(center [0] - 1)
        p_down = int(center [0])

    if s%2 != 0:   
        center.append(int(s/2)+1)
        p_left  = int(center[1] - 1)
        p_right = int(center [1] - 1)
    else:
        center.append(int(s/2))
        p_left  = int(center[1] - 1)
        p_right = int(center[1])
    
    zero_pad_img1 = cv2.copyMakeBorder(img1,p_up,p_down,p_left,p_right,cv2.BORDER_CONSTANT,value=[0,0,0])#top bottom left right
    zero_pad_img2 = cv2.copyMakeBorder(img2,p_up,p_down,p_left,p_right,cv2.BORDER_CONSTANT,value=[0,0,0])#top bottom left right  
    center_pt = (center[0]-1)*window_size[1]+ center[1]
    return zero_pad_img1, zero_pad_img2, center, center_pt  


def make_disparity_map_right_to_left(img1,img2,window_size, center, center_pt):
    #For every pixel get a window
    height_1 ,width_1, channel1 = img1.shape
    height_2 ,width_2, channel2 = img2.shape
    disparity = []
    temp = []
    s_range = 35
    output_image = np.zeros((height_2,width_2 + window_size[0]),np.uint8)
    for y in range (height_1):
        cy = y + center[1]
        for x in range (width_1):
            cx = x + center[0]
            window_1 = []
            for j in range (window_size[0]):
                for i in range (window_size[1]):
                    m  = x + i
                    n  = y + j
                    if (m<width_1) and (n<height_1):
                        px_c1 = int(img1[ n , m , 0 ])
                        px_c2 = int(img1[ n , m , 1 ])
                        px_c3 = int(img1[ n , m , 2 ])
                        px = px_c1 + px_c2 + px_c3
                        window_1.append(px)
                        q =img1[n,m]
                        temp.append(q)
                        #if m == cx and n == cy:
                           #print('Pixel of Interest W1 : ',temp[center_pt-1], ' Located At  : ', cy , cx )
            factor = int(255/(s_range - 1))
            #print( 'Length of window 1' , len(window_1))
            disparity_x = calculate_disparity_right_to_left( img2 , window_size , center, center_pt, window_1 , s_range, y , x)
            disparity.append(disparity_x)
            #print(' Disparity ' , disparity_x )
            output_image [ y , cx ] = disparity_x * factor
    #height_3 ,width_3, channel3 = output_image.shape
    #print(height_3 ,width_3, channel3)

    output_image_2 = np.zeros((height_2,width_2 - s_range),np.uint8)
    h2, w2 = output_image_2.shape
    for y in range (h2):
        for x in range(w2):
            output_image_2[y,x] = output_image[y, x + window_size[0] ]
    return output_image_2, factor

def calculate_disparity_right_to_left(  img2 , window_size , center, center_pt, w1 , s_range , y2 , x2 ):
    factor_avg = window_size[1] * window_size[0]
    height_2 ,width_2, channel2 = img2.shape
    poi = []
    sum_diff = []
    past_y = y2
    cy = y2 + center[1]
    corr_x = x2
    center_pointer = corr_x + center[0]
    for x2 in range (corr_x,(s_range + corr_x)): # for 10 pixels to the right of pixel in window 1
        cx = x2 + center[0]
        windows_2 = []
        diffs = []
        temp = []
        for j in range (window_size[0]): #calculate 10 windows for image2 one at a time 
            for i in range (window_size[1]):
                m  = x2 + i
                n  = y2 + j
                if (m < s_range + center_pointer and m < width_2 -1 and n < height_2 - 1  ):
                    px_c1 = int(img2[ n , m , 0 ])
                    px_c2 = int(img2[ n , m , 1 ])
                    px_c3 = int(img2[ n , m , 2 ])
                    px = px_c1 + px_c2 + px_c3
                    windows_2.append(px)#give 9 points of window 2
                    q = img2[n , m]
                    temp.append(q)
                    #if m == cx and n == cy:
                       #print('Pixel of Interest W2 : ',temp[center_pt-1], ' Located At  : ' , cy  , cx )

        #------------------------------------------------------------SAD---------------------------------------------------------------------------------         
        for k in range (len(windows_2)):#calculate difference of each element for single window 1 and window 2 for all 10 windows one at a time
            #print ( 'cx' , cx )
            if cx < (width_2) :
                #print('Element w1 at ' , k , ' = ' , int(w1[k]))
                #print('Element w2 ' , k , ' = ' , int(windows_2[k]) )
                ith = int(w1[k])
                jth = int(windows_2[k])
                ji =  abs(jth - ith)
                #print('Difference' , ji)
                diffs.append(ji)

        sum_of_window_difference = sum(diffs)#calculate sum of differences of two single windows for all 10 windows one at a time
        sum_of_window_difference_mean = sum_of_window_difference / factor_avg
        #print('Sum of window difference ',sum_of_window_difference)
        sum_diff.append(sum_of_window_difference_mean)
        
    match_location = sum_diff.index(min(sum_diff)) # find location of pixel where difference is minimum for all 10 windows at once
    #print('Minimuim Difference' ,  min(sum_diff) ,' At Location ', match_location + center_pointer )
    corresponding_x =  match_location + center_pointer
    #print ('Corresponding x pixel ' , corresponding_x , 'for x = ' , center_pointer , 'and y' , cy)

    '''
    if corresponding_x <= width_2 and cy <= height_2:
       disparity_x =  corresponding_x - center_pointer
    else:
       disparity_x = 0
    disparity_x =  corresponding_x - center_pointer
    '''

    disparity_x =  corresponding_x - center_pointer
    return disparity_x                         

def make_disparity_map_left_to_right(img1,img2,window_size, center, center_pt):
    #For every pixel get a window
    height_1 ,width_1, channel1 = img1.shape
    height_2 ,width_2, channel2 = img2.shape
    disparity = []
    temp = []
    s_range = 35
    output_image = np.zeros((height_2,width_2 + window_size[0]),np.uint8)
    for y in reversed (range (height_1)):
        cy = y - center[1]
        for x in reversed (range (width_1)):
            cx = x - center[0]
            window_1 = []
            for j in range (window_size[0]):
                for i in range (window_size[1]):
                    m  = x - i
                    n  = y - j
                    if (m>0) and (n>0):
                        px_c1 = int(img1[ n , m , 0 ])
                        px_c2 = int(img1[ n , m , 1 ])
                        px_c3 = int(img1[ n , m , 2 ])
                        px = px_c1 + px_c2 + px_c3
                        #print(px)
                        window_1.append(px)
                        q =img1[n,m]
                        temp.append(q)
                        #if m == cx and n == cy:
                           #print('Pixel of Interest W1 : ',temp[center_pt-1], ' Located At  : ', cy , cx )
            #print( 'Length', len(window_1))          
            factor = int(255/(s_range - 1))
            #print( 'Length of window 1' , len(window_1))
            disparity_x = calculate_disparity_left_to_right( img2 , window_size , center, center_pt, window_1 , s_range, y , x) #calculate by searching in direction
            disparity.append(disparity_x)
            #print(' Disparity ' , disparity_x )
            output_image [ y , cx ] = disparity_x * factor
    #height_3 ,width_3, channel3 = output_image.shape
    #print(height_3 ,width_3, channel3)
    
    output_image_2 = np.zeros((height_2,width_2 - s_range),np.uint8)
    h2, w2 = output_image_2.shape
    for y in reversed(range (h2 - 1)):
        x_display = w2 - 1
        for x in reversed(range(s_range , width_2 - 1)):
            #print(x_display,x)
            output_image_2[y,x_display] = output_image[y, x - window_size[0]]
            x_display = x_display - 1 
    return output_image_2

def calculate_disparity_left_to_right(  img2 , window_size , center, center_pt, w1 , s_range , y2 , x2 ):
    factor_avg = window_size[1] * window_size[0]
    height_2 ,width_2, channel2 = img2.shape
    poi = []
    sum_diff = []
    past_y = y2
    cy = y2 - center[1]
    corr_x = x2
    center_pointer = corr_x - center[0]
    #print(cy ,  corr_x)
    for x2 in reversed (range ((corr_x - s_range),corr_x)): # for 10 pixels to the right of pixel in window 1
        cx = x2 - center[0]
        windows_2 = []
        diffs = []
        temp = []
        for j in range (window_size[0]): #calculate 10 windows for image2 one at a time 
            for i in range (window_size[1]):
                m  = x2 - i
                n  = y2 - j
                #if (m < s_range + center_pointer and m > 0 and n > 0):
                if (m > 0)and (n > 0):
                    px_c1 = int(img2[ n , m , 0 ])
                    px_c2 = int(img2[ n , m , 1 ])
                    px_c3 = int(img2[ n , m , 2 ])
                    px = px_c1 + px_c2 + px_c3
                    #print(px)
                    windows_2.append(px)#give 9 points of window 2
                    q = img2[n , m]
                    temp.append(q)
                    #if m == cx and n == cy:
                       #print('Pixel of Interest W2 : ',temp[center_pt-1], ' Located At  : ' , cy  , cx )

        #------------------------------------------------------------SAD---------------------------------------------------------------------------------         
        for k in range (len(windows_2)):#calculate difference of each element for single window 1 and window 2 for all 10 windows one at a time
            #print ( 'cx' , cx )
            if cx > 0 :
                #print('Element w1 at ' , k , ' = ' , int(w1[k]))
                #print('Element w2 ' , k , ' = ' , int(windows_2[k]) )
                ith = int(w1[k])
                jth = int(windows_2[k])
                ji =  abs(jth - ith)
                #print('Difference' , ji)
                diffs.append(ji)

        sum_of_window_difference = sum(diffs)#calculate sum of differences of two single windows for all 10 windows one at a time
        sum_of_window_difference_mean = sum_of_window_difference / factor_avg
        #print('Sum of window difference ',sum_of_window_difference_mean)
        sum_diff.append(sum_of_window_difference_mean)
        
    match_location = sum_diff.index(min(sum_diff)) # find location of pixel where difference is minimum for all 10 windows at once
    #print('Minimuim Difference' ,  min(sum_diff) ,' At Location ', match_location + center_pointer )
    corresponding_x =  match_location + center_pointer
    #print ('Corresponding x pixel ' , corresponding_x , 'for x = ' , center_pointer , 'and y' , cy)
    disparity_x =  corresponding_x - center_pointer
    return disparity_x                         

def validity_check(r_limg,l_rimg, factor):
    h, w = r_limg.shape
    h2, w2 = l_rimg.shape
    #print('For Right to left scan  ', h , w)
    #print('For Left to Right scan ' , h2 , w2)
    output_validity = np.zeros((h2,w2),np.uint8)
    for y in range (h):
        for x in range(w):
            if r_limg[y,x] - l_rimg[y,x] <= 5 * factor:
            #if r_limg[y,x] - l_rimg[y,x] == 0 :   
               output_validity[y,x] = r_limg[y,x]
            else:
               output_validity[y,x] = 0
    return output_validity

 
def main():
    start = time.time()
    # Read both images
    #create_window('Image 1')
    #create_window('Image 2')
    img1 = cv2.imread('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Test Images\\cone_img1.png')
    img2 = cv2.imread('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Test Images\\cone_img2.png')
    window_size = [5,5]
    
    # Zero pad both images
    pad_img1, pad_img2, center, center_pt  = pad(img1,img2,window_size)
    #show_image('Image 1', pad_img1)
    #show_image('Image 2', pad_img2)

    # find sum of absolute differences
    r_limg, factor = make_disparity_map_right_to_left(pad_img1,pad_img2,window_size, center , center_pt)
    l_rimg = make_disparity_map_left_to_right(pad_img2,pad_img1,window_size, center , center_pt)

    cv2.imshow('Right to Left Image' , r_limg)
    cv2.imshow('Left to Right Image' , l_rimg)

    validity = validity_check(r_limg, l_rimg, factor)
    cv2.imshow('Validity', validity)
    end = time.time()
    print('Execution Time' , (end - start)/60 , 'minutes')

if __name__ == "__main__": main()
