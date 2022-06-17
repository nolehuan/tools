import os
import random
import uuid
import numpy as np
import cv2
import math

center = (3.0, 4.0)
angle = 3.0
scale = 2.0
R =cv2.getRotationMatrix2D(center, angle, scale)
print(R)
angle = angle * math.pi / 180
M = np.array([
[   scale * math.cos(angle), scale * math.sin(angle), (1 - scale * math.cos(angle)) * center[0] - scale * math.sin(angle) * center[1] ],
[ - scale * math.sin(angle), scale * math.cos(angle), (1 - scale * math.cos(angle)) * center[1] + scale * math.sin(angle) * center[0] ]])
print(M)

x = np.full((2, 3), 7, dtype=np.uint8)
print(x)

idx = [0, 2, 4, 6]
for i, j in enumerate(idx):
    print(i, j)

print("{:06}".format(1))

print(np.max((0, 1)))

x = tuple(["s", "d", "f"])
print(x)

print(os.path.dirname(os.path.dirname(__file__)))

print(os.getenv("HOME", None))

# x = range(10)[:12]
x = range(10)[:5]

x = np.zeros((2, 3))
print(x)

x = np.random.randint(1,3,3)
print(x.shape)
x = np.expand_dims(x, 1)
x = np.expand_dims(x, 1)
print(x.shape)
x = np.hstack((x, x))
print(x.shape)

print(np.random.uniform(-1, 1, 3))
print(np.random.randint(0, 2, 3))

seed = uuid.uuid4().int % 2**32
print(seed)
random.seed(seed)
print(seed)

size = (3, 6)
x = random.randint(*size)
print(x)

print(os.path.realpath(__file__))
print(os.path.split(os.path.realpath(__file__)))
print(__file__)

os.makedirs("./tmp", exist_ok=True)

print(".".join(["s", "s"]))

print(os.path.dirname("./files/times.txt"))
print(os.path.basename("./files/times.txt"))

s = "a_b-c"
s = s.replace("-", "_")
print(s)