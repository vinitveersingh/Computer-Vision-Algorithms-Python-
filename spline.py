import cv2
import numpy as np

GAUSSIAN_MASK_IMG1 = []
GAUSSIAN_MASK_IMG2 = []
LAPLACIAN_IMAGE1 = []
LAPLACIAN_IMAGE2 = []
RECONSTRUCT = []
  
#----------------------------------------------------------------UTILITY FUNCTIONS--------------------------------------------------------------------------
def create_window(window_name):
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

def destroy_window(window_name):
    cv2.destroyWindow(window_name,cv2.WINDOW_AUTOSIZE)

def show_image(window_name,img):
    cv2.imshow(window_name,img)


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

#----------------------------------------------------------------MAKE GAUSSIAN MASKS  ----------------------------------------------------------
def create_mask(img,img2):
    create_window('Original Image 1')
    show_image('Original Image 1',img)
    cv2.setMouseCallback('Original Image 1',region_pt)
    create_window('Original Image 2')
    show_image('Original Image 2',img2)

def region_pt(event,x,y,flags,params):
    global rect
    if event == cv2.EVENT_LBUTTONDOWN:
        rect = [(x,y)]
    if event == cv2.EVENT_LBUTTONUP:
        rect.append((x,y))
        draw_rect(rect,img)

def draw_rect(rect,img):
    if (len(rect)!=0):
        img2= img.copy()
        cv2.rectangle(img2,rect[0],rect[1],(255,0,0),2)
        print('x pt1',rect[0][0],'y pt1',rect[0][1],'xy pt1',rect[0])
        print('x pt1',rect[1][0],'y pt1',rect[1][1],'xy pt1',rect[1])
        print('height',rect[1][1]-rect[0][1])
        print('width',rect[1][0]-rect[0][0])
        x1 = rect[0][0]
        y1 = rect[0][1]
        x2 = rect[1][0]
        y2 = rect[1][1]
        r_height = rect[1][1]-rect[0][1]
        r_width = rect[1][0]-rect[0][0]
        make_mask(x1,y1,x2,y2,img)

              
def make_mask(x1,y1,x2,y2,img):
    global GAUSSIAN_MASK_IMG1
    global GAUSSIAN_MASK_IMG2
    height,width, channel = img.shape
    mask=np.ones((height,width,3),np.uint8)
    inverted_mask= np.zeros((height,width,3),np.uint8)
    for c in range(channel):
        for h in range (y1,y2+1):
            for w in range (x1,x2+1): 
                inverted_mask[h,w,c]+=1
                
    for c in range(channel):
        for h in range (y1,y2+1):
            for w in range (x1,x2+1): 
                mask[h,w,c]*=0    

    GAUSSIAN_MASK_IMG1 = get_gp_mask(mask)
    GAUSSIAN_MASK_IMG2 = get_gp_mask(inverted_mask)

    blend_mask_laplacian()


def get_gp_mask(img1):
# generate Gaussian pyramid for image
    G = img1.copy()
    gaussianpyramids_img1 = [img1]
    for i in range(2):   
        G = cv2.pyrDown(G)
        gaussianpyramids_img1.append(G)
        if i == 3:
            break
    return gaussianpyramids_img1


#----------------------------------------------------------------MAKE LAPLACIAN OF IMAGES----------------------------------------------------------
def get_gp_images(img1):
# generate Gaussian pyramid for image
    G = img1.copy()
    gaussianpyramids_img1 = [img1]
    for i in range(2):   
        G = cv2.pyrDown(G)
        gaussianpyramids_img1.append(G)
        if i == 3:
            break
        
    laplacian_levels = get_lp(gaussianpyramids_img1)
    return laplacian_levels

def get_lp(gaussianpyramids_img1):
    laplacianpyramids_img1 = []
    for j in range(len(gaussianpyramids_img1)):
        if j+1 < len(gaussianpyramids_img1):
            laplacianpyramids_elements= cv2.subtract(gaussianpyramids_img1[j], cv2.pyrUp(gaussianpyramids_img1[j+1]))
            laplacianpyramids_img1.append(laplacianpyramids_elements)
        
        else:
            laplacianpyramids_element_final= gaussianpyramids_img1[j]
            laplacianpyramids_img1.append(laplacianpyramids_element_final)

    
    return laplacianpyramids_img1


def create_laplacians(img1,img2):
    global LAPLACIAN_IMAGE1
    global LAPLACIAN_IMAGE2
    LAPLACIAN_IMAGE1 = get_gp_images(img1)
    LAPLACIAN_IMAGE2 = get_gp_images(img2)
    
    '''
    for s in range(len(LAPLACIAN_IMAGE1)):
        print(s)
        windowname = 'Laplacian level  Image 1 ' + str(s) 
        cv2.imshow(windowname,LAPLACIAN_IMAGE1[s]) 

    for q in range(len(LAPLACIAN_IMAGE2)):
        print(q)
        windowname = 'Laplacian level Image 2 ' + str(q) 
        cv2.imshow(windowname,LAPLACIAN_IMAGE2[q])
    '''
    
#----------------------------------------------------------------BLEND LAPLACIAN AND MASK----------------------------------------------------------
def blend_mask_laplacian():


    global RECONSTRUCT
    print(len(LAPLACIAN_IMAGE1))
    print(len(LAPLACIAN_IMAGE2))
    print(len(GAUSSIAN_MASK_IMG1))
    print(len(GAUSSIAN_MASK_IMG1))

    for r in range(len(LAPLACIAN_IMAGE1)):
        blend_mask_laplacian1 = np.multiply(LAPLACIAN_IMAGE1[r],GAUSSIAN_MASK_IMG1[r])
        blend_mask_laplacian2 = np.multiply(LAPLACIAN_IMAGE2[r],GAUSSIAN_MASK_IMG2[r])
        #img_final = cv2.add(blend_mask_laplacian1,blend_mask_laplacian2)
        img_final = np.add(blend_mask_laplacian1,blend_mask_laplacian2)
        RECONSTRUCT.append(img_final)

    reconstruct(RECONSTRUCT,RECONSTRUCT[len(RECONSTRUCT)-1])
    
def reconstruct(laplacianpyramids_img1,laplacianpyramids_element_final):
    reconstructed_gaussian_level = laplacianpyramids_element_final
    reconstructed_gaussian_img1 = [reconstructed_gaussian_level]

    display_counter= 0
    reconstructed_counter = 0 
    for k in reversed(range(len(laplacianpyramids_img1)-1)):
        reconstructed_gaussian_level = laplacianpyramids_img1[k] + cv2.pyrUp(reconstructed_gaussian_img1[reconstructed_counter])
        reconstructed_counter+=1
        reconstructed_gaussian_img1.append(reconstructed_gaussian_level)
        if reconstructed_counter == len(laplacianpyramids_img1):
            break
    
    window_name= 'Blended Image' + str(display_counter)
    display_counter+=1
    cv2.imshow(window_name,reconstructed_gaussian_img1[len(reconstructed_gaussian_img1)-1]) 
    

img_path1 = input('Enter Image 1 Path : ')
#img_path = 'C:\\Users\\Vinit\\Desktop\\lenna.png'
img = cv2.imread(img_path1)
img_path2 = input('Enter Image 2 Path : ')
img2 = cv2.imread(img_path2)

#img = cv2.imread('C:\\Users\\Vinit\\Desktop\\apple.jpg')
#img2= cv2.imread('C:\\Users\\Vinit\\Desktop\\orange.jpg')

create_laplacians(img,img2)
create_mask(img,img2)



