# 读取图片，base64 编码

# import os
import glob
# from PIL import Image
import base64
# from io import BytesIO
# import matplotlib.pyplot as plt

# picture_dir=os.path.join(os.getcwd(),'*.jpg')
picture_dir = './*.jpg'
for jpgfile in glob.glob(picture_dir):
    # encode()
    file = open(jpgfile,'rb')
    base64_data = base64.b64encode(file.read())
    print(base64_data)
    # decode()
    # byte_data = base64.b64decode(base64_data)
    # image_data = BytesIO(byte_data)
    # img = Image.open(image_data)
    # #show pioture
    # plt.imshow(img)
    # plt.show()
