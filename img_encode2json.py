# 图片 base64 编码，输出到 json 文件

import json, base64
import copy
import glob

picture_dir = './*.jpg'
for jpgfile in glob.glob(picture_dir):
    # encode()
    file = open(jpgfile,'rb')
    base64_data = base64.b64encode(file.read())
    # print(base64_data)

with open("./*.json",'r') as temp:
    annotation_dict = json.load(temp)
temp.close()

annotation_dict['imageData'] = base64_data.decode()
annotation_dict['shapes'][0]['label'] = "white_solid"
annotation_dict['shapes'][0]['points'] = [[16.77922077922078,524.9090909090909],
        [114.18181818181817,498.93506493506493],
        [250.54545454545453,463.87012987012986],
        [350.54545454545456,440.49350649350646],
        [433.6623376623377,419.7142857142857],
        [531.0649350649351,398.93506493506493],
        [631.0649350649351,371.6623376623377],
        [716.7792207792207,349.5844155844156],
        [785.6103896103896,331.4025974025974],
        [841.4545454545454,313.2207792207792],
        [854.4415584415584,305.42857142857144],
        [854.4415584415584,293.7402597402597]]

dictionary = copy.deepcopy(annotation_dict['shapes'][0])
annotation_dict['shapes'].append(dictionary) # 浅拷贝

annotation_dict['shapes'][1]['label'] = "white_solid"
annotation_dict['shapes'][1]['points'] = [[2298.5974025974024,987.2467532467533],
        [2115.4805194805194,865.1688311688312],
        [1929.7662337662337,741.7922077922078],
        [1763.5324675324675,640.4935064935065],
        [1638.857142857143,561.2727272727273],
        [1529.7662337662337,495.038961038961],
        [1425.8701298701299,435.2987012987013],
        [1347.948051948052,392.4415584415584],
        [1279.1168831168832,356.0779220779221],
        [1229.7662337662337,334.0],
        [1193.4025974025974,317.1168831168831],
        [1158.3376623376623,304.12987012987014],
        [1124.5714285714284,298.93506493506493],
        [1089.5064935064934,295.038961038961],
        [1049.2467532467533,287.24675324675326],
        [1015.4805194805194,285.94805194805195],
        [992.1038961038961,285.94805194805195]]

with open("./*.json",'w') as f:
    json.dump(annotation_dict, f)
f.close()