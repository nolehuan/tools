import numpy as np

file_in = '../dataset/KITTI/raw/road/2011_09_26_drive_0015/2011_09_26/2011_09_26_drive_0015_sync/image_02/timestamps.txt'

stamps = []
with open(file_in, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        stamp = line.split(' ')
        if len(stamp) == 2:
            stamps.append(stamp[-1][6:])
f.close()

t0 = stamps[0]

file_out = './times.txt'

with open(file_out, 'w') as f:
    for stamp in stamps:
        f.write(str(float(stamp) - float(t0)))
        f.write('\n')
f.close()
