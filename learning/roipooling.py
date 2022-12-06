# https://github.com/Parsa33033/RoiPooling

import numpy as np

class RoiPooling():
    def __init__(self, mode='tf', pool_size=(7,7)):
        """
        tf: (height, width, channels)
        th: (channels, height, width)
        """
        self.mode = mode
        self.pool_size = pool_size

    def pool(self, region):
        """
        region: roi fetched from feature map
        return: tf (1, height, width, channels)
                th (1, channels, height, widt)
        """
        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode == 'th':
            region_channels, region_height, region_width = region.shape
            pool = np.zeros((region_channels, pool_height, pool_width))
        h_step = region_height / pool_height
        w_step = region_width / pool_width

        for i in range(pool_height):
            for j in range(pool_width):
                x_min = int(j * w_step)
                x_max = int((j + 1) * w_step)
                y_min = int(i * h_step)
                y_max = int((i + 1) * h_step)

                if x_min == x_max or y_min == y_max:
                    continue
                if self.mode == 'tf':
                    pool[i, j, :] = np.max(region[y_min : y_max, x_min : x_max, :], axis=(0,1))
                elif self.mode == 'th':
                    pool[:, i, j] = np.max(region[:, y_min : y_max, x_min : x_max], axis=(1,2))
        return pool

    def get_region(self, feature_map, roi_dimensions):
        """
        feature_map: (1, height, width, channels)
        roi_dimensions: a region of interest dimensions
        """
        x_min, y_min, x_max, y_max = roi_dimensions
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        if self.mode == 'tf':
            r = np.squeeze(feature_map)[y_min : y_max, x_min : x_max, :]
        elif self.mode == 'th':
            r = np.squeeze(feature_map)[:, y_min : y_max, x_min : x_max]
        return r

    def get_pooled_rois(self, feature_map, roi_batch):
        """
        roi_batch: usually is 256 for faster rcnn
        """
        pool = []
        for region_dim in roi_batch:
            region = self.get_region(feature_map, region_dim)
            p = self.pool(region)
            pool.append(p)
        return np.array(pool)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    mode = 'th'
    h = 50
    w = 38
    if mode == 'tf':
        feature_map = np.zeros((h, w, 512))
    elif mode == 'th':
        feature_map = np.zeros((512, h, w))
    for i in range(h):
        for j in range(w):
            if np.random.rand() < 0.1: # [0, 1)均匀分布
                if mode == 'tf':
                    feature_map[i, j, :] = np.random.rand()
                elif mode == 'th':
                    feature_map[:, i, j] = np.random.rand()
    roi_batch = np.array([[0,0,10,10],[2,2,5,5]])

    roi_pooled = RoiPooling(mode=mode).get_pooled_rois(feature_map, roi_batch)

    if mode == 'tf':
        _, ax = plt.subplots(2)
        ax[0].imshow(feature_map[..., 0])
        x_min, y_min, x_max, y_max = roi_batch[0]
        ax[0].add_patch(patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                        edgecolor='r', facecolor='none', linewidth=1))
        ax[1].imshow(roi_pooled[0, ..., 0])
        plt.show()
    elif mode == 'th':
        _, ax = plt.subplots(2)
        ax[0].imshow(feature_map[0, ...])
        x_min, y_min, x_max, y_max = roi_batch[0]
        ax[0].add_patch(patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                        edgecolor='r', facecolor='none', linewidth=1))
        ax[1].imshow(roi_pooled[0, 0, ...])
        plt.show()


