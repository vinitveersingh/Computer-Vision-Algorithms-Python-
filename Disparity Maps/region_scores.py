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
    
    zero_pad_img1 = cv2.copyMakeBorder(img1,p_up,p_down,p_left,p_right,cv2.BORDER_CONSTANT,value=[0,0,0])
    zero_pad_img2 = cv2.copyMakeBorder(img2,p_up,p_down,p_left,p_right,cv2.BORDER_CONSTANT,value=[0,0,0])
    center_pt = (center[0]-1)*window_size[1]+ center[1]
    return zero_pad_img1, zero_pad_img2, center, center_pt  


def make_disparity_map(img1,img2,window_size, center, center_pt , s_range, method):
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
                        
            factor = int(255/(s_range - 1))
            disparity_x = calculate_disparity( img2 , window_size , center, center_pt, window_1 , s_range, y , x , method)
            disparity.append(disparity_x)
            output_image [ y , cx ] = disparity_x * factor

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
    cy = y2 + center[1]
    corr_x = x2
    center_pointer = corr_x + center[0]
    for x2 in range (corr_x,(s_range + corr_x)):
        cx = x2 + center[0]
        windows_2 = []
        diffs = []
        temp = []
        for j in range (window_size[0]):
            for i in range (window_size[1]):
                m  = x2 + i
                n  = y2 + j
                if (m < s_range + center_pointer and m < width_2 -1 and n < height_2 - 1  ):
                    px_c1 = int(img2[ n , m , 0 ])
                    px_c2 = int(img2[ n , m , 1 ])
                    px_c3 = int(img2[ n , m , 2 ])
                    px = px_c1 + px_c2 + px_c3
                    windows_2.append(px)
                    q = img2[n , m]
                    temp.append(q)
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
    img1 = cv2.imread('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Test Images\\art_img1.png')
    img2 = cv2.imread('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Test Images\\art_img2.png')
    h,w,c = img1.shape
    print('For image of size: ', h , 'x' , w)
    window = int(input('Enter window size: ' ))
    s_range = int(input('Enter search range: '))
    method = int(input('Select method: \n1.SAD \n2.SSD \n3.NCC\n'))
    window_size = [window]
    window_size.append(window)
    pad_img1, pad_img2, center, center_pt  = pad(img1,img2,window_size)
    l_rimg = make_disparity_map(pad_img1,pad_img2,window_size, center , center_pt , s_range , method)
    cv2.imshow('LR Image', l_rimg)
    cv2.imwrite('C:\\Users\\Vinit\\Desktop\\Computer Vision\\Project 2\\Region  Based\Results\\art_region_ssd.png',l_rimg)
    end = time.time()
    print(' Execution Time:', end - start)

if __name__ == "__main__": main()
