import cv2
import numpy as np
import matplotlib.pyplot as plt

def modified_otsu(img):
    threshold = 0
    h, w = img.shape
    n_pix = np.zeros((256, ))
    pix_prob_dis = np.zeros((256, ))

    for i in range (h):
        for j in range (w):
            n_pix[img[i, j]] += 1

    for k in range(256):
        pix_prob_dis[k] = 1.0 * n_pix[k] / (h * w - n_pix[0])

    var_max = 0.0
    for m in range(1, 256):
        w_fg = 0.0
        w_bg = 0.0
        var_tmp = 0.0
        avg_fg = 0.0
        avg_bg = 0.0
        avg_fg_tmp = 0.0
        avg_bg_tmp = 0.0
        for n in range(1, 256):
            if (n <= m):
                w_bg += pix_prob_dis[n]
                avg_bg_tmp += n * pix_prob_dis[n]
            else:
                w_fg += pix_prob_dis[n]
                avg_fg_tmp += n * pix_prob_dis[n]

        if (w_fg == 0 or w_bg == 0): continue
        avg_fg = avg_fg_tmp / w_fg
        avg_bg = avg_bg_tmp / w_bg
        var_tmp = w_fg * w_bg * pow(avg_fg - avg_bg, 2)
        if (var_tmp > var_max):
            var_max = var_tmp
            threshold = m

    return threshold




# png_file = '../4/4_image_roi.png'
png_file = '../1/_image_roi.png'
# img = cv2.imread(png_file, 0)
img = cv2.imread(png_file, -1)
print(img.shape)
# cv2.imshow("demo", img)
# cv2.waitKey(0)
# plt.imshow(img, cmap='gray')
# plt.show()

# img = np.asarray(img)
# minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
# print(maxVal)
# print(maxLoc)
# while maxVal > 0:
#     img[np.where(img == maxVal)] = 0
#     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
#     print(maxVal)
#     print(maxLoc)
threshold = modified_otsu(img)
print(threshold)

# # bins = np.arange(256)
# hist = np.histogram(img, bins=256)
# print(hist[0].shape)
# print(hist[0])
# print(hist[1].shape)
# print(hist[1])

# plt.hist(img.ravel(), bins=256, rwidth=0.8, range=(0,255))
# plt.show()

retval, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
plt.imshow(img, cmap='gray')
plt.show()

