import numpy as np

# x = np.array([1,2,3,4])
# print(x)
# print(x.shape)
# y = x[..., None]
# print(y)
# print(y.shape)


# npfile = np.load('../r2d2/imgs/0000000000.png.r2d2')

# print(npfile['imsize'])
# print(npfile['keypoints'][0])
# print(npfile['descriptors'][0])
# print(npfile['scores'].shape)


# npfile = np.load('./kitti_09_ape/distances.npy')
# npfile = np.load('./kitti_09_ape/distances_from_start.npy')
npfile = np.load('./files/kitti_09_ape/alignment_transformation_sim3.npy')
# npfile = np.load('./kitti_09_ape/error_array.npy')
# npfile = np.load('./kitti_09_ape/seconds_from_start.npy')
# npfile = np.load('./kitti_09_ape/timestamps.npy')

print(npfile)
