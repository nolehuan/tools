import os
import random
import uuid
import numpy as np

np.random.uniform(-1, 1, 3)


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