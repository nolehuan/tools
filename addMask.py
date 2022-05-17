import cv2
import glob

label_files = '../LoFTR/assets/odometry/09/json/*/label.png'
for label_path in glob.glob(label_files):
    name = label_path.split('\\')[-2][:-5]
    label = cv2.imread(label_path, -1)
    image_path = '../LoFTR/assets/odometry/09/image_2/' + name + '.png'
    image = cv2.imread(image_path, -1)
    masked_image = cv2.addWeighted(image, 1.0, label, 0.8, 0.0)
    # cv2.imshow("demo", masked_image)
    # cv2.waitKey(0)
    cv2.imwrite('../LoFTR/assets/odometry/09/masked/' + name + '.png', masked_image)

