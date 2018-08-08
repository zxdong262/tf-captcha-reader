'''
seperate single letters from captcha into several images

use openCV.findContours
'''
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from scipy.misc import bytescale
from copy import deepcopy
from math import pow

TEXT_IMAGE_SIZE = (28, 28)
EDGE = 5
SMALL = 3

def getMinMax(arr):
  '''
  get x,y's max/min
  '''
  xx = []
  yy = []
  for i in range(len(arr)):
    a = arr[i]
    xx.append(a[0][0])
    yy.append(a[0][1])
  maxX = np.max(xx)
  minX = np.min(xx)
  maxY = np.max(yy)
  minY = np.min(yy)

  return (maxX, minX, maxY, minY)

def getExternalBoxPoints(contour):
  '''
  get external box of one contour
  '''
  maxX, minX, maxY, minY = getMinMax(contour)

  return [
    [[minX, minY]],
    [[minX, maxY]],
    [[maxX, maxY]],
    [[maxX, minY]]
  ]

def sortShapes(shape):
  '''
  sort shapes by size
  @return width^2 + height^2
  '''
  width = shape[2][0][0] - shape[0][0][0]
  height = shape[2][0][1] - shape[0][0][1]
  return pow(width, 2) + pow(height, 2)

def removeSmallShapesForce(shapes):
  '''
  force remove the small ones
  '''
  res = []
  for i in range(len(shapes)):
    shape = shapes[i]
    height = shape[2][0][1] - shape[0][0][1]
    if height > SMALL:
      res.append(shape)
  if len(res) < len(shapes):
    print(
      'force remove small ones:', len(shapes) - len(res)
    )
  return res

def removeSmallShapes(shapes, count):
  '''
  remove extra unrecognized shapes,
  remove the small ones
  '''
  s = deepcopy(shapes)
  s = sorted(s, key=sortShapes)
  s = s[count:]
  s = sorted(s, key=lambda c: shapes.index(c))
  return s

def splitShape(shape):
  '''
  horizontally equal split one shape to two shape
  '''
  minX = shape[0][0][0]
  maxX = shape[2][0][0]
  hx = int(maxX/2 + minX/2)
  minY = shape[0][0][1]
  maxY = shape[2][0][1]
  return [
    [
      [[minX, minY]],
      [[minX, maxY]],
      [[hx, maxY]],
      [[hx, minY]]
    ],
    [
      [[hx, minY]],
      [[hx, maxY]],
      [[maxX, maxY]],
      [[maxX, minY]]
    ]
  ]

def splitShapes(shapes, count):
  '''
  when shapes is not equal to text count,
  split big images
  '''
  s = deepcopy(shapes)
  s = sorted(s, key=sortShapes, reverse=True)
  s = s[0:count]
  res = deepcopy(shapes)
  for i in range(len(s)):
    a = s[i]
    idx = res.index(a)
    splited = splitShape(a)
    del res[idx]
    res.insert(idx, splited[0])
    res.insert(idx + 1, splited[1])

  return res

def getExternalBoxs(contours, charCount = None):
  '''
  get external boxs of contours
  '''
  res = []
  for i in range(len(contours) - 1):
    a = contours[i]
    ps = getExternalBoxPoints(a)
    res.append(ps)
  res = removeInnerBox(res)
  res = concatShapes(res)
  #length = len(res)
  res = removeSmallShapesForce(res)
  if charCount and len(res) > charCount:
    print('remove small shapes')
    res = removeSmallShapes(res, len(res) - charCount)
  elif charCount and len(res) < charCount:
    print('splitShapes')
    res = splitShapes(res, charCount - len(res))

  return res

def columCheck(t, t0):
  '''
  check if t is the same column with t0
  '''
  maxX = t[0]
  minX = t[1]
  maxX0 = t0[0]
  minX0 = t0[1]
  if (t == t0):
    return False
  return  (
    np.abs(maxX - maxX0) < EDGE and
    np.abs(minX - minX0) < EDGE
  )

def isSameColum(a, arr):
  '''
  check if list a is the same column with other shape
  '''
  try:
    arr.index(a)
  except:
    return -1
  t = getMinMax(a)
  for i in range(len(arr)):
    aa = arr[i]
    t0 = getMinMax(aa)
    if columCheck(t, t0):
      return i
  return False

def concatShapes(arr):
  '''
  concat two shape basicly in same column
  '''
  res = []
  narr = deepcopy(arr)
  for i in range(len(arr)):
    a = arr[i]
    idx = isSameColum(a, narr)
    if idx == -1:
      continue
    elif not idx:
      res.append(a)
    else:
      res.append(
        getExternalBoxPoints(a + narr[idx])
      )
      narr.pop(idx)
  return res

def insideCheck(t, t0):
  '''
  check if t is inside t0
  '''
  maxX, minX, maxY, minY = t
  maxX0, minX0, maxY0, minY0 = t0
  if (t == t0):
    return False
  return  (
    maxX <= maxX0 and maxX >= minX0 and
    minX <= maxX0 and minX >= minX0 and
    maxY <= maxY0 and maxY >= minY0 and
    minY <= maxY0 and minY >= minY0
  )

def isInner(a, arr):
  '''
  check if list a is innner shape
  '''
  t = getMinMax(a)
  for i in range(len(arr)):
    aa = arr[i]
    t0 = getMinMax(aa)
    if insideCheck(t, t0):
      return True
  return False

def removeInnerBox(arr):
  '''
  remove all inner shapes
  '''
  res = []
  for i in range(len(arr)):
    a = arr[i]
    if not isInner(a, arr):
      res.append(a)
  return res

def imageGroup(image, charCount=None, shouldSaveExample=False):
  '''
  use openCV.findContours to seprate charactors
  '''
  if shouldSaveExample:
    image.save('example-captcha.png')

  img8 = np.array(image.convert('L'))
  img8[img8 != 255] = 0

  if shouldSaveExample:
    x1 = Image.fromarray(img8)
    x1.save('example-binary.png')

  imgCopy = deepcopy(img8)

  c = cv2.findContours(img8, mode=1, method=2)
  c01 = c[1]
  c01 = getExternalBoxs(c01, charCount)

  if shouldSaveExample:
    c1 = np.array(c01)
    copy1 = deepcopy(imgCopy)
    print(len(c1))
    cv2.drawContours(copy1, c1, -1, (67,189,66), 1)
    x2 = Image.fromarray(copy1)
    x2.save('example-findContours.png')

  res = []

  for i in range(len(c01)):
    shape = c01[i]
    copy2 = deepcopy(img8)
    im2 = Image.fromarray(copy2)
    x = shape[0][0][0]
    y = shape[0][0][1]
    w = shape[2][0][0]
    h = shape[2][0][1]
    box = (x, y, w, h)
    print(box)
    im2 = im2.crop(box)
    im2.load()
    res.append(im2)
    if shouldSaveExample:
      im2.save('example-split-' + str(i) + '.png')

  return res