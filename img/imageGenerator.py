'''
captcha image generator
'''
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from imageGrouping import imageGroup

def buildDic(arr):
  dicc = {}
  dicc1 = {}
  for i in range(len(arr)):
    ii = arr[i]
    cc = chr(ii)
    dicc[i] = cc
    dicc1[cc] = i
  return dicc, dicc1

randint = np.random.randint

CHAR_POOL = list(range(97, 123)) + list(range(65, 91)) + list(range(48, 58))
CHAR_DIC, CHAR_INDEX_DIC = buildDic(CHAR_POOL)
IMAGE_SIZE = (160, 36)
TEXT_IMAGE_SIZE = (28, 28)



def randomChar():
  '''
  return one random char in [a-zA-Z0-9]
  '''
  count = len(CHAR_POOL)
  n = randint(0, count)
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

def createTextImg():
  '''
  create image with random char.
  '''
  BG_COLOR = (255, 255, 255, 0)
  R = randint(60, 190)
  G = randint(60, 190)
  B = randint(60, 190)
  rotate = randint(0, 13)
  TEXT_COLOR = (R, G, B)

  TEXT_POS = (0, 0)
  fontSize = 26

  img = Image.new('RGBA', TEXT_IMAGE_SIZE, color = BG_COLOR)
  font = ImageFont.truetype('./node_modules/open-sans-fonts/open-sans/Regular/OpenSans-Regular.ttf', size=fontSize)
  d = ImageDraw.Draw(img)
  char = randomChar()
  d.text(TEXT_POS, char, fill=TEXT_COLOR, font=font)
  img = img.rotate(rotate, expand=0)
  return img, char

def computePastePos(j):
  '''
  compute paste postion
  '''
  start = -16
  if j == 0:
    start = 5
  left = randint(start, 8) + j * (TEXT_IMAGE_SIZE[0] + 4)
  top = randint(0, 4)
  return (left, top)

def createImg(i):
  '''
  create random captcha image with .
  '''
  BG_COLOR = 'white'
  textCount = randint(4, 6)
  img = Image.new('RGBA', IMAGE_SIZE, color = BG_COLOR)
  txt = ''
  for j in range(textCount):
    textImg, char = createTextImg()
    txt = txt + char
    img.paste(textImg, box=computePastePos(j), mask=textImg)

  return img, txt

def main():
  img = createImg(0)
  image, txt = img
  print(txt)
  imageGroup(image, charCount=len(txt), shouldSaveExample=True)

if __name__ == '__main__':
  main()