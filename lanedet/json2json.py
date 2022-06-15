# 修改 json 文件内容

import json, glob
from collections import OrderedDict
from tqdm import tqdm

json_files = '../FCLane/group0_json/*.json'
with tqdm(total = 820) as pbar:
    for jsonfile in glob.glob(json_files):
        with open(jsonfile,'r',encoding="utf-8") as temp:
            annotation_dict = json.load(temp, object_pairs_hook=OrderedDict)
        temp.close()

        name = jsonfile.split('\\')[-1][:-5]
        annotation_dict['imageData'] = None

        json_out = '../FCLane/group0_json_without_imagedata/' + name + '.json'
        with open(json_out,'w',encoding="utf-8") as f:
            json.dump(annotation_dict, f)
        f.close()
        pbar.update(1)

# cnt = 0
# json_files = '../label/group0_0/*.json'
# for jsonfile in glob.glob(json_files):
#     with open(jsonfile,'r') as temp:
#         annotation_dict = json.load(temp)
#     temp.close()

#     name = jsonfile.split('\\')[-1][:-5]
#     if annotation_dict['imagePath'] == "template.jpg":
#         annotation_dict['imagePath'] = name + '.jpg'
#         with open(jsonfile,'w') as f:
#             json.dump(annotation_dict, f)
#         f.close()
#         cnt += 1
# print(cnt)

# json_files = '../group0/*.json'
# for jsonfile in glob.glob(json_files):
#     with open(jsonfile,'r') as temp:
#         annotation_dict = json.load(temp)
#     temp.close()

#     name = str(int(jsonfile.split('\\')[-1][:-5]) + 9).zfill(6)
#     annotation_dict['imagePath'] = name + '.jpg'
#     img_path = '../label/group3/' + name + '.jpg'
#     file = open(img_path,'rb')
#     img_base64 = base64.b64encode(file.read())
#     annotation_dict['imageData'] = img_base64.decode()

#     jsoncopy = '../label/group3/' + name + '.json'
#     with open(jsoncopy,'w') as f:
#         json.dump(annotation_dict, f)
#     f.close()