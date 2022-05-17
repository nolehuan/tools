#########################################################################
##  Data augmentation
##  PINet
##  2002.06604
#########################################################################

import math
import numpy as np
import cv2
import random
from copy import deepcopy

#########################################################################
## some iamge transform utils
#########################################################################
def Translate_Points(point,translation): 
    point = point + translation 
    
    return point

def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


#########################################################################
## Data augmentation class
#########################################################################
class Data_augmentation(object):
    #################################################################################################################
    ## initialize
    #################################################################################################################
    def __init__(self):
        self.x_size = 960
        self.y_size = 640
        img = np.zeros((self.y_size,self.x_size,3), np.uint8)
        self.Flip(img)
        self.Translation(img)
        self.Rotate(img)
        self.Gaussian(img)
        self.Change_intensity(img)
        self.Shadow(img)

    #################################################################################################################
    ## Add Gaussian noise
    #################################################################################################################
    def Gaussian(self, input_img):
        img = np.zeros((self.y_size,self.x_size,3), np.uint8)
        m = (0,0,0)
        s = (20,20,20)
        
        test_image = deepcopy(input_img)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)
        cv2.randn(img,m,s)
        test_image = test_image + img
        # test_image =  np.rollaxis(test_image, axis=2, start=0)
        output_img = test_image
        return output_img

    #################################################################################################################
    ## Change intensity
    #################################################################################################################
    def Change_intensity(self, input_img):
        test_image = deepcopy(input_img)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)

        hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        value = int(random.uniform(-60.0, 60.0))
        if value > 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = -1*value
            v[v < lim] = 0
            v[v >= lim] -= lim                
        final_hsv = cv2.merge((h, s, v))
        test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)
        output_img = test_image
        return output_img

    #################################################################################################################
    ## Generate random shadow in random region
    #################################################################################################################
    def Shadow(self, input_img,  min_alpha=0.5, max_alpha = 0.75):
        test_image = deepcopy(input_img)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)
        # test_image =  np.rollaxis(test_image, axis=2, start=0)

        top_x, bottom_x = np.random.randint(0, self.x_size, 2) # low, high, size
        coin = 0
        rows, cols, _ = test_image.shape
        shadow_img = test_image.copy()
        if coin == 0:
            rand = np.random.randint(2)
            vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32) # 角顶; 顶点; 至高点
            if rand == 0:
                vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
            elif rand == 1:
                vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
            mask = test_image.copy()
            channel_count = test_image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (0,) * channel_count
            cv2.fillPoly(mask, [vertices], ignore_mask_color)
            rand_alpha = np.random.uniform(min_alpha, max_alpha)
            cv2.addWeighted(mask, rand_alpha, test_image, 1 - rand_alpha, 0., shadow_img)
            # shadow_img =  np.rollaxis(shadow_img, axis=2, start=0)
            output_img = shadow_img
            return output_img

    #################################################################################################################
    ## Flip
    #################################################################################################################
    def Flip(self, input_img):
        temp_image = deepcopy(input_img)
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)

        temp_image = cv2.flip(temp_image, 1) # 1 水平翻转 0 垂直翻转 -1 水平垂直翻转
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)
        output_img = temp_image
        return output_img

    #################################################################################################################
    ## Translation
    #################################################################################################################
    def Translation(self, input_img):
        temp_image = deepcopy(input_img)
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)       

        tx = np.random.randint(-50, 50)
        ty = np.random.randint(-30, 30)

        temp_image = cv2.warpAffine(temp_image, np.float32([[1,0,tx],[0,1,ty]]), (self.x_size, self.y_size)) # 仿射变换
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)
        output_img = temp_image
        return output_img

    #################################################################################################################
    ## Rotate
    #################################################################################################################
    def Rotate(self, input_img):
        temp_image = deepcopy(input_img)
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)  

        angle = np.random.randint(-10, 10)

        M = cv2.getRotationMatrix2D((self.x_size//2,self.y_size//2),angle,1) # 旋转中心，旋转角度，缩放因子

        temp_image = cv2.warpAffine(temp_image, M, (self.x_size, self.y_size))
        # temp_image =  np.rollaxis(temp_image, axis=2, start=0)
        output_img = temp_image
        return output_img

img = cv2.imread(r"D:/Users/buaa/Desktop/classification/data/train/0/FILE200420-174156F_744.jpg")
img = cv2.resize(img, (960, 640), interpolation=cv2.INTER_LINEAR)

data_aug = Data_augmentation()
# img = data_aug.Shadow(img)
# img = data_aug.Flip(img)
# img = data_aug.Gaussian(img)
# img = data_aug.Rotate(img)
# img = data_aug.Translation(img)
img = data_aug.Change_intensity(img)

cv2.imshow("img", img)
cv2.waitKey(0)