# captcha-reader
demo read captcha image with TensorFlow

## process

image generated:(4 or 5 random en char or digit, multi-color, random roate 0 ~ 4 degree)

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
# (testData, testLabels) = createData(2000)

Epoch 25/25
36029/36029 [==============================] - 2s 54us/step - loss: 0.0752 - acc: 0.9669
8970/8970 [==============================] - 0s 29us/step
evaluate test data set:
test_loss: 0.28423821404674937
test_acc: 0.9385730211817168
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




