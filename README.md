# captcha-reader
demo read captcha image with TensorFlow

## process

image generated:(4 or 5 char, multi-color, roate)

![example-captcha.png](example-images/example-captcha.png)

make it binary:

![example-binary.png](example-images/example-binary.png)

get shapes with opencv.findContours:

![example-findContours.png](example-images/example-findContours.png)

split it:
![example-split-0.png](example-images/example-split-0.png)
![example-split-1.png](example-images/example-split-1.png)
![example-split-2.png](example-images/example-split-2.png)
![example-split-3.png](example-images/example-split-3.png)

then feed to tensorflow.keras

result:
```bash
# (trainData, trainLabels) = createData(8000)
# (testData, testLabels) = createData(1000)

Epoch 25/25
35948/35948 [==============================] - 2s 53us/step - loss: 0.0743 - acc: 0.9673
4492/4492 [==============================] - 0s 30us/step
evaluate test data set:
test_loss: 0.22218923419142264
test_acc: 0.949020480854853
```

it can be improved by add more train data or add more epochs.

## run
```bash

# install Helper libraries, for ubuntu 16.04 only
sudo apt-get install python3-pip python3-tk
pip3 install matplotlib numpy scipy matplotlib ipython jupyter pandas sympy nose --user

# clone the repo
git clone git@github.com:zxdong262/captcha-reader.git
cd captcha-reader
npm i
python3 main.py

```




