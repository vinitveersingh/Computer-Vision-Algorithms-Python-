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

def Harris(imcolor):
    
    gray_image = cv2.cvtColor(imcolor, cv2.COLOR_BGR2GRAY)
    height_1, width_1, channel_1 = imcolor.shape
    dst = cv2.cornerHarris(gray_image,2,3,0.04)
    dst = cv2.dilate(dst,None)
    imcolor[dst>0.000001*dst.max()]=[128,128,128]
    return imcolor

def pad_harris( img1 , window_size):
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

    return zero_pad_img1 
    
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
    zero_pad_img2 = cv2.copyMakeBorder(img2,p_up,p_down,p_left,p_right ,cv2.BORDER_CONSTANT,value=[0,0,0])#top bottom left right  
    center_pt = (center[0]-1)*window_size[1]+ center[1]
    return zero_pad_img1, zero_pad_img2, center, center_pt  


def make_disparity_map(img1,img2,window_size, center, center_pt, img_harris_final, s_range, method):
    factor2 = window_size[0]* window_size[1] 
    #For every pixel get a window
    height_1 ,width_1, channel1 = img1.shape
    height_2 ,width_2, channel2 = img2.shape
    disparity = []
    temp = []
    output_image = np.zeros((height_2,width_2 + window_size[0]),np.uint8)
    for y in range (height_1):
        cy = y + center[1]
        for x in range (width_1):
            cx = x + center[0]
            window_1 = []
            if img_harris_final[y,x,0] == 128 and img_harris_final[y,x,1] == 128 and img_harris_final[y,x,2] == 128:
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
                        if m == cx and n == cy:
                           center_saved_x = cx
                           center_saved_y = cy
                           #print('Pixel of Interest W2 : ',temp[center_pt-1], ' Located At  : ' , center_saved_y  , center_saved_x )
                        if len(window_1) == factor2:
                           factor = int(255/(s_range - 1))
                           #print( 'Length of window 1' , len(window_1))
                           disparity_x = calculate_disparity( img2 , window_size , center, center_pt, window_1 , s_range, center_saved_y , center_saved_x, method)
                           #print(' Disparity ' , disparity_x )
                           output_image [ y , center_saved_x ] = disparity_x * (factor/4)

    output_image_2 = np.zeros((height_2,width_2 - s_range),np.uint8)
    h2, w2 = output_image_2.shape
    for y in range (h2):
        for x in range(w2):
            output_image_2[y,x] = output_image[y, x + window_size[0] ]

    return output_image_2


def calculate_disparity(  img2 , window_size , center, center_pt, w1 , s_range , y2 , x2 , method ):
    factor_avg = window_size[1] * window_size[0]
    height_2 ,width_2, channel2 = img2.shape
    poi = []
    sum_diff = []
    past_y = y2
    cy = y2 
    corr_x = x2
    center_pointer = corr_x
    for x2 in range (corr_x,(s_range + corr_x)): # for 10 pixels to the right of pixel in window 1
        cx = x2 
        windows_2 = []
        diffs = []
        temp = []
        for j in range (window_size[0]): #calculate 10 windows for image2 one at a time 
            for i in range (window_size[1]):
                m  = x2 + i
                n  = y2 + j
                if  m < s_range + center_pointer and m < width_2 -1 and n < height_2 - 1 :
                    px_c1 = int(img2[ n , m , 0 ])
                    px_c2 = int(img2[ n , m , 1 ])
                    px_c3 = int(img2[ n , m , 2 ])
                    px = px_c1 + px_c2 + px_c3
                    windows_2.append(px)#give 9 points of window 2
                    q = img2[n , m]
                    temp.append(q)
                    #if m == cx and n == cy:
                       #print('Pixel Located At  : ' , cy  , cx )

        #print( 'Length of window 1' , len(windows_2))

        if method == 1 :
            diffs = SAD(windows_2, cx, width_2, w1)
        elif method == 2 :
            diffs = SSD(windows_2, cx, width_2, w1)
        elif method == 3:
            diffs = NCC(windows_2, cx, width_2, w1, factor_avg)
            
        if method == 1 or method == 2:
            sum_of_window_difference = sum(diffs)
            sum_of_window_difference_mean = sum_of_window_difference / factor_avg
            sum_diff.append(sum_of_window_difference_mean)
        elif method ==3:
            sum_of_window_difference = sum(diffs)#calculate sum of differences of two single windows for all 10 windows one at a time
            #print('Sum of window difference ',sum_of_window_difference)
            sum_diff.append(sum_of_window_difference)
            
    if method == 1 or method == 2:         
        match_location = sum_diff.index(min(sum_diff))
        corresponding_x =  match_location + center_pointer
        disparity_x =  corresponding_x - center_pointer
    elif method == 3:
        match_location = sum_diff.index(max(sum_diff)) # find location of pixel where difference is minimum for all 10 windows at once
        corresponding_x =  match_location + center_pointer
        disparity_x =  corresponding_x - center_pointer
    return disparity_x                         


def SAD(windows_2, cx, width_2, w1):
    diffs = []
    for k in range (len(windows_2)):
        if cx < (width_2) :
            ith = int(w1[k])
            jth = int(windows_2[k])
            ji =  abs(jth - ith)
            diffs.append(ji)
    return diffs

def SSD(windows_2, cx, width_2, w1):
    diffs = []
    for k in range (len(windows_2)):
        if cx < (width_2) :
            ith = int(w1[k])
            jth = int(windows_2[k])
            ji =  jth - ith
            square_diff_ji = math.pow(ji,2) 
            diffs.append(square_diff_ji)       
    return diffs

def NCC(windows_2, cx, width_2, w1, factor_avg):
    diffs = []
    for k in range (len(windows_2)):
        if cx < (width_2) :           
            avg_window1 = sum(w1) / factor_avg
            avg_window2 = sum(windows_2) / factor_avg
            ith = int(w1[k]) - avg_window1
            jth = int(windows_2[k]) - avg_window2
            sqrt_i = math.sqrt( math.pow(ith, 2))
            sqrt_j = math.sqrt( math.pow(jth, 2))
            if sqrt_i and sqrt_j > 0:
                ncc1 = ith / sqrt_i
                ncc2 = jth / sqrt_j
                ncc= ncc1 * ncc2
                diffs.append(ncc)
    return diffs

                   
def main():
    start = time.time()
    window_size = []
    img1 = cv2.imread('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Test Images\\rain_img1.png')
    img2 = cv2.imread('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Test Images\\rain_img2.png')
    h,w,c = img1.shape
    print('For image of size: ', h , 'x' , w)
    window = int(input('Enter window size: ' ))
    s_range = int(input('Enter search range: '))
    method = int(input('Select method: \n1.SAD \n2.SSD \n3.NCC\n'))
    window_size = [window]
    window_size.append(window)
    
    img1cp =img1.copy()
    pad_img1, pad_img2, center, center_pt  = pad(img1,img2,window_size)
    img1_harris = Harris(img1cp)
    img_harris_final = pad_harris( img1_harris , window_size )
    
    cv2.imshow('Harris' , img_harris_final)
 
    # find sum of absolute differences
    l_rimg = make_disparity_map(img_harris_final,pad_img2,window_size, center , center_pt, img_harris_final , s_range, method)
    cv2.imshow('Feature Images' , l_rimg)
    cv2.imwrite('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Feature Based\\feature_based_raindeer_ssd.png',l_rimg)
    end = time.time()
    print('Execution Time:', end - start)
    
if __name__ == "__main__": main()
