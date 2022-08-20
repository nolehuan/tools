import os
import random
from collections import Counter
import uuid
import numpy as np
import cv2
import math
import io
import itertools
import datetime
import scipy.special
import re
import copy




# a = [1, 2]
# a = [1, 2, [3, 4]]
a = [1, 2, {"a": 1}]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
a[0] = 2
print(a, id(a))
print(b, id(b))
print(c, id(c))
print(d, id(d))

a = "copy"
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
print(id(a))
print(id(b))
print(id(c))
print(id(d))

a = [0, "a", 1, [], False, None]
print([bool(i) for i in a])
print(any(a))
print(all(a))

dict = {"one": 1, "two": 2}
# del dict["one"]
dict.pop("one")
print(dict)

def func(k, v, dict = {}):
    dict[k] = v
    print(dict)
func("one", 1)
func("two", 2)
func("three", 3, {})

a = "%.03f" % 1.3335
print(a, type(a))
b = round(float(a), 2)
print(b)

a = b"hello"
b = "world".encode()
print(a, type(a))
print(b, type(b))

a = [1, 3, 5]
b = "asdf"
c = (a, b, a)
r = [i for i in zip(a, b, c)]
print(r)

for i in range(1, 5, 1):
    print(i)

x = "abc"
y = "def"
z = ["g", "h", "i"]
print(x.join(z))
print(x.join(y))

s = "nnn 88mmm"
r = re.sub(r"\d+", "100", s)
print(r)

s = "<a>lh</a><a>lh</a>"
r = re.findall('<a>(.*)</a>', s)
print(r) # 贪婪匹配，尽可能多匹配
r = re.findall('<a>(.*?)</a>', s)
print(r) # 非贪婪匹配

s = "正则 404 not found -1.2 3.4"
l = s.split(" ")
r = re.findall('-*\d+\.?\d*|[a-zA-Z]+', s)
# \d+ 匹配数字 | 连接多个匹配方式 [a-zA-Z] 匹配单词
# -*\d+\.?\d* 匹配小数
for i in r:
    if i in l:
        l.remove(i)
s = " ".join(l)
print(s)

s = '<div class="name">china</div>'
ret = re.findall(r'<div class=".*">(.*?)</div>', s)
print(ret)

l = [[1,2],[8,9]]
ll = np.array(l).flatten().tolist()
print(ll)

try:
    a = 0
    raise Exception(a)
except Exception as e:
    print(e)
finally:
    print("over")

s = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
d = datetime.datetime.now().isoweekday()
print(s, d)

a = 0
b = 1
a, b = b, a
print(a, b)

l = [1,2,3,4,5]
def func(x):
    return x % 2
ll = filter(func, l)
l = [x for x in ll]
print(l)

s = "ajldjlajfdljfddd"
ret = Counter(s)
print(ret)

d = {"name":"lh", "age":24, "city":"rz"}
l = sorted(d.items(), key=lambda kv : kv[0], reverse=False)
print(l)

s = "ajldjlajfdljfddd"
s = set(s)
s = list(s)
s.sort(reverse=False)
s = "".join(s)
print(s)

a = 1
b = 1
print(id(a))
print(id(b))

x = random.randint(0, 3) # 随机整数
y = np.random.randn(5) # 随机小数
z = random.random() # 0-1 小数
print(x)
print(y)
print(z)

l = [1, 2, 3]
def func(x):
    return x**2
ml = map(func, l)
print(list(ml))

class bike(object):
    def __init__(self, color, wheel) -> None:
        print("init")
        print(color)
        print(wheel)
        self.color = color
        self.wheel = wheel
    def __new__(cls, color, wheel):
        print("new")
        return super().__new__(cls)
bike("red", 4)

def func(*args, **kwargs):
    for key in args:
        print(key)
    for key, value in kwargs.items():
        print(key, value)

func('a', 'b', 'c', name="banban", age=24)

# noi
class Truth:
    pass
a = Truth()
print(bool(a))

b = (1, [2, 4])
b[1].extend([6, 8])
print(b)

idx = np.arange(3) + 1
print(idx)

out = np.array([[[0,1,2,3],[4,5,6,7],[8,9,10,11]], [[12,13,14,15],[16,17,18,19],[20,21,22,23]]])
print(out)
print(out.shape)
out = out[:, ::-1, :]
print(out)
out_argmax = out.argmax(0)
print(out_argmax)
print(out_argmax.shape)
out = scipy.special.softmax(out[:-1, :, :], axis=0)
print(out)

x = np.array([[2, 4, 5, 1], [3, 4, 9, 0]])
print(x.shape)
print(x.argmax(1))

x = tuple((1, 2))
print(x)
x = list(x)
print(x)

def test():
    return 1, 2, 3
*_, x = test()
print(x)

x = np.array([1,2,2,3,3,4])
print(np.where(x[:-1] == x[1:]))

x = np.linspace(0, 5, 6, endpoint=True)
print(x)
x = np.linspace(0, 5, 6, endpoint=False)
print(x)

x = np.empty((0, 5))
print(x)

print(datetime.timedelta(seconds=100))

def test(a, b, c):
    print(a)
    print(b)
    print(c)
d = {'a':'A','b':'B','c':'C'}
test(*d)
test(**d)

# cv2.Scharr()

# l = [1, 2, 3]
l = [[1, 2, 3], [4, 5, 6]]
print(l)
print(*l)

res = itertools.zip_longest('abc', '12')
print(res)
for x in res:
    print(x)

x = list([1, 2, 3, 4, 5, 6])
print(x[1::2])

s = io.StringIO()
s.write("hello")
print(s.getvalue())

x = 0.02
print(f"xx {x:.3f}")

x = np.array([1])
print(x)
print(x.item())

for maindir, subdir, file_name_list in os.walk("./vision/"):
    print(maindir)
    print(subdir)
    for filename in file_name_list:
        print(filename)
        path = os.path.join(maindir, filename)
        print(os.path.splitext(path))

print(os.cpu_count())

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