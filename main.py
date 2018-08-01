'''
simple single color image with text recognition
'''

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from tensorflow import keras
from functools import reduce

def buildDic(arr):
  dicc = {}
  dicc1 = {}
  for i in range(len(arr)):
    ii = arr[i]
    cc = chr(ii)
    dicc[i] = cc
    dicc1[cc] = i
  return dicc, dicc1

CHAR_POOL = list(range(97, 123)) + list(range(65, 91)) + list(range(48, 58))
CHAR_DIC, CHAR_INDEX_DIC = buildDic(CHAR_POOL)
SIZE = (30, 30)

def randomChar():
  '''
  return random char [a-zA-Z0-9]
  '''
  count = len(CHAR_POOL)
  n = np.random.randint(0, count)
  return chr(CHAR_POOL[n])

def rgb2int(arr):
  '''
  convert rgb color array to int
  eg: [r, g, b] => 65536 * r + 256 * g + b
  '''
  R = arr[0]
  G = arr[1]
  B = arr[2]
  return R * 299/1000 + G * 587/1000 + B * 114/1000

# def reshapeArray(arr):
#   '''
#   reshape data array, convert all rgb array to int
#   '''
#   x = len(arr)
#   y = len(arr[0])
#   z = len(arr[0][0])
#   res = []
#   def mapper(x):
#     return np.array(
#       map(mapper1, x)
#     )
#   def mapper(x):
#     return np.array(
#       map(mapper1, x)
#     )
#   return np.array(
#     map(mapper, arr)
#   )

def createImg(i):
  '''
  create random captcha image dataarray.
  '''
  BG_COLOR = (0, 0, 0)
  TEXT_COLOR = (255, 255, 255)

  TEXT_POS = (np.random.randint(1, 10), np.random.randint(1, 10))
  fontSize = 18

  img = Image.new('RGB', SIZE, color = BG_COLOR)
  font = ImageFont.truetype('./node_modules/open-sans-fonts/open-sans/Regular/OpenSans-Regular.ttf', size=fontSize)
  d = ImageDraw.Draw(img)
  char = randomChar()
  d.text(TEXT_POS, char, fill=TEXT_COLOR, font=font)
  arr = np.array(img)
  arr = arr.reshape((SIZE[0] * SIZE[1], 3))
  arr1 = []
  for x in range(len(arr)):
    a = arr[x]
    arr1.append(
      rgb2int(a)
    )
  arr1 = np.array(arr1) / 255.0
  arr1 = arr1.reshape(SIZE)
  arr1 = arr1

  if i == 0:
    img.save('example.png')
  return (arr1, CHAR_INDEX_DIC[char])

def createData(n):
  '''
  create data and labels array, with length = n
  '''
  data = []
  labels = []
  for i in range(n):
    (arr, char) = createImg(i)
    data.append(arr)
    labels.append(char)
  return (np.array(data), np.array(labels))

def main():
  '''main'''

  print('tensorflow version:', tf.__version__)

  (trainData, trainLabels) = createData(1000)
  (testData, testLabels) = createData(124)

  model = keras.Sequential([
    keras.layers.Flatten(input_shape=SIZE),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(CHAR_POOL), activation=tf.nn.softmax)
  ])

  model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  model.fit(trainData, trainLabels, epochs=40)

  test_loss, test_acc = model.evaluate(testData, testLabels)
  print(test_loss, test_acc)


if __name__ == '__main__':
  main()