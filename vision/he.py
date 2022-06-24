import numpy as np
import cv2

def hist_equalization(image):
    output = np.copy(image)
    h, w = image.shape
    total_pixel = h * w
    prob = []

    for i in range(256):
        prob.append(np.sum(image == i) / total_pixel)

    prob_t = 0
    prob_output = []
    for j in range(256):
        prob_t += prob[j]
        prob_output.append(prob_t)
    
    for k in range(256):
        output[np.where(image == k)] = 255 * prob_output[k]
    
    return output

if __name__ == "__main__":
    image = cv2.imread("./files/0000000000.png", 0)

    dst = np.zeros_like(image)
    cv2.equalizeHist(image, dst)

    output = hist_equalization(image)

    output = np.vstack([image, dst, output])

    cv2.imshow("demo", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

