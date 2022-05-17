# generate txt file

# file = open('../FCLane/FCLane_train_gt.txt', 'w')
# for i in range(434): # 820
#     pic_path = "group0/" + str(3 * i).zfill(6) + ".jpg laneseg_label/group0/" + str(3 * i).zfill(6) + ".png"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(128):
#     pic_path = "group1/" + str(10000 + 3 * i).zfill(6) + ".jpg laneseg_label/group1/" + str(10000 + 3 * i).zfill(6) + ".png"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(76):
#     pic_path = "group2/" + str(20000 + 3 * i).zfill(6) + ".jpg laneseg_label/group2/" + str(20000 + 3 * i).zfill(6) + ".png"
#     file.write(pic_path)
#     file.write("\n")
# file.close()

# file = open('../FCLane/FCLane_train.txt', 'w')
# for i in range(434): # 820
#     pic_path = "group0/" + str(3 * i).zfill(6) + ".jpg"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(128):
#     pic_path = "group1/" + str(10000 + 3 * i).zfill(6) + ".jpg"
#     file.write(pic_path)
#     file.write("\n")
# for i in range(76):
#     pic_path = "group2/" + str(20000 + 3 * i).zfill(6) + ".jpg"
#     file.write(pic_path)
#     file.write("\n")
# file.close()

file = open('./eigen_test_files_with_gt.txt', 'w')
for i in range(297):
    img_path = "2011_09_26_drive_0015/2011_09_26/2011_09_26_drive_0015_sync/image_02/data/" + str(i).zfill(10) + ".png None 721.5377"
    file.write(img_path)
    file.write("\n")