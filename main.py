'''
captcha reader demo

captcha with 4 or 5 char, random color for every char, random rotate some degree, see `example-images/example-captcha.png`
make it binary, see `example-images/example-binary.png`
use opencv findcontontours to cut out every char image, see `example-images/example-split-*.png`
then use tensorflow to train and read the test images

'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from img.imageGenerator import createImg, TEXT_IMAGE_SIZE, CHAR_POOL, CHAR_INDEX_DIC, CHAR_DIC
from img.imageGrouping import imageSplit

def rgb2int(arr):
  '''
  convert rgb color array to int
  eg: [r, g, b] => 65536 * r + 256 * g + b
  '''
  R = arr[0]
  G = arr[1]
  B = arr[2]
  return R * 299/1000 + G * 587/1000 + B * 114/1000

def convertToDataArray(img, i, j, ch):
  '''
  convert image to data array
  and resize to 28*28
  '''
  BG_COLOR = (255, 255, 255)
  base = Image.new('RGB', TEXT_IMAGE_SIZE, color = BG_COLOR)
  img.convert('RGB')
  size = img.size
  left = int((TEXT_IMAGE_SIZE[0] - size[0]) / 2)
  top = int((TEXT_IMAGE_SIZE[1] - size[1]) / 2)
  if left < 0: left = 0
  if top < 0: top = 0
  base.paste(img, box=(left, top))
  shouldSave = i < 1
  if shouldSave:
    img.save(
      'example-resized-char-b' + str(i) +
      '-' + str(j) + '-' + ch + '.png'
      )
    base.save(
      'example-resized-char' + str(i) +
      '-' + str(j) + '-' + ch +'.png'
      )
  arr = np.array(base)
  arr = arr.reshape((TEXT_IMAGE_SIZE[0] * TEXT_IMAGE_SIZE[1], 3))
  arr1 = []
  for x in range(len(arr)):
    a = arr[x]
    arr1.append(
      rgb2int(a)
    )
  arr1 = (255 - np.array(arr1)) / 255.0
  arr1 = arr1.reshape(TEXT_IMAGE_SIZE)
  return arr1


def createData(n):
  '''
  create data and labels array, with length = n
  '''
  data = []
  labels = []
  for i in range(n):
    (img, text) = createImg(i)
    tlist = list(text)
    le = len(tlist)
    shouldSave = i == 0
    imgs = imageSplit(img, charCount=le, shouldSaveExample=shouldSave)
    for j in range(len(imgs)):
      im = imgs[j]
      data.append(
        convertToDataArray(im, i, j, tlist[j])
      )
    tlist = list(
      map(lambda x: CHAR_INDEX_DIC[x], tlist)
    )
    # if shouldSave:
    #   print(tlist, data)
    labels = labels + tlist
  return (np.array(data), np.array(labels))

def main():
  '''main'''

  print('tensorflow version:', tf.__version__)

  print('loading data')
  (trainData, trainLabels) = createData(8000)
  (testData, testLabels) = createData(2000)

  print('trainData:', len(trainData))
  print('testData:', len(testData))

  model = keras.Sequential([
    keras.layers.Flatten(input_shape=TEXT_IMAGE_SIZE),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(CHAR_POOL), activation=tf.nn.softmax)
  ])

  model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )

  model.fit(
    trainData,
    trainLabels,
    epochs=25
  )

  test_loss, test_acc = model.evaluate(testData, testLabels)
  print('evaluate test data set:')
  print('test_loss:', test_loss)
  print('test_acc:', test_acc)

  print('predict example-image(example-images/example-captcha.png):')
  with Image.open('example-images/example-captcha.png') as img:
    imgs = imageSplit(img)
    data = []
    i = 200
    for j in range(len(imgs)):
      im = imgs[j]
      arr = convertToDataArray(im, i, j, 'unknown')
      arr = np.expand_dims(arr, 0)
      predictions = model.predict(arr)
      prediction = predictions[0]
      prediction = np.argmax(prediction)
      char = ''
      try:
        char = CHAR_DIC[prediction]
      except:
        char = ''
      data.append(char)
    print(data)

if __name__ == '__main__':
  main()