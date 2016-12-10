import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = None
img2 = None
right_clicks_img1 = []
right_clicks_img2 = []
spoints = []
affine_pts1 = []

'''  Create and destroy windows '''
def destroy_window(window_name):
    cv2.destroyWindow(window_name)

def create_window(windows_name):
    cv2.namedWindow(windows_name,cv2.WINDOW_AUTOSIZE)

''' Get x,y coordinates on mouse_click'''
def extract_point(x,y,image_name):
    if image_name == 'Image_Point_1':
       right_clicks_img1.append([x, y,1])
       print(x,y)
       print ( img1[x,y,0])
    if image_name == 'Image_Point_2':
       right_clicks_img2.append([x, y,1])

''' Final Mouse callback function plot points multiple times on mouse click'''       
def plot_points(event, x, y, flags,affine_transformation_matrix):
    if event ==1:
        create_window('Image 2') 
        destroy_window('Image 2')
        spoints.append([x,y,1])
        for i in range(len(spoints)):
            selected_point=np.asmatrix(spoints[i]).transpose()
            print(affine_transformation_matrix)
            plot_point=np.dot(affine_transformation_matrix,selected_point)
            create_window('Image 2') 
            cv2.circle(img2,(plot_point[0],plot_point[1]),3,(0,0,255),-1);#plot circle as point
            cv2.imshow('Image 2',img2)

            
''' Final Mouse callback function to extract points'''              
def mouse_callback(event, x, y, flags, params):
    if event == 1:
        extract_point(x,y,params)

        
def calculate_matching_points(affine_transformation_matrix):
    cv2.imshow('Image 1',img1)
    cv2.setMouseCallback('Image 1', plot_points ,affine_transformation_matrix)


''' Get affine matrix after extracting points'''       
def calculate_affine_matrix():
    if(len(right_clicks_img1)== len(right_clicks_img2)) and (len(right_clicks_img1) > 2) :
        A= np.asmatrix(right_clicks_img1).transpose();
        AT= np.asmatrix(right_clicks_img1)#my input is read as transposed matrix
        B=np.asmatrix(right_clicks_img2).transpose();
        affine_transformation_matrix = np.dot(np.dot(B, AT), np.linalg.inv(np.dot(A, AT)))
        destroy_window('Image 2')
        print('Affine paramater using ',A.shape[1],'points',affine_transformation_matrix)
        print('Click on image 1 to get other corresponding points on image 2 ')
        calculate_matching_points(affine_transformation_matrix)
        
    else:
        print("Invalid Number of Points")
        cv2.destroyAllWindows()

#Program Starts Here
img1 = cv2.imread('C:\\Users\\Vinit\\Desktop\\1.jpg')
create_window('Image 1')
img2 = cv2.imread('C:\\Users\\Vinit\\Desktop\\2.jpg')
create_window('Image 2')    
cv2.imshow('Image 1',img1) 
cv2.imshow('Image 2',img2)
cv2.setMouseCallback('Image 1', mouse_callback, 'Image_Point_1')#opencv fuction that works on mouse click event.Parameters(imgae name, callback function name,parameter name)
cv2.setMouseCallback('Image 2', mouse_callback, 'Image_Point_2')
waitk = cv2.waitKey(0) & 0xFF
if waitk == ord('c'):#press c to execute code
    calculate_affine_matrix()


''''
1. Program starts here
2. create windows
3. mouse_callback function to extract points
4. press c to calculate affine matrix
5. affine matrix calls calculate_matching_points which is a hlper function to plot points 
'''
