
# tf-captcha-reader

Read captcha image with TensorFlow. [Tutorial](https://html5beta.com/page/tutorial-reading-captcha-with-tensorflow.html)

## Work flow

Image generated:(4 or 5 random en char or digit, multi-color, random roate 0 ~ 4 degree)

![example-captcha.png](example-images/example-captcha.png)

Make it binary:

![example-binary.png](example-images/example-binary.png)

Get shapes with opencv.findContours:

![example-findContours.png](example-images/example-findContours.png)

Split it:
![example-split-0.png](example-images/example-split-0.png)
![example-split-1.png](example-images/example-split-1.png)
![example-split-2.png](example-images/example-split-2.png)
![example-split-3.png](example-images/example-split-3.png)

Then feed to tensorflow.keras

Result:
```bash
# (trainData, trainLabels) = createData(8000)
# (testData, testLabels) = createData(2000)

Epoch 25/25
36029/36029 [==============================] - 2s 54us/step - loss: 0.0752 - acc: 0.9669
8970/8970 [==============================] - 0s 29us/step
evaluate test data set:
test_loss: 0.28423821404674937
test_acc: 0.9385730211817168
predict example-image(example-images/example-captcha.png):
['l', '8', 'V', 'T']
```

It can be improved by add more train data or add more epochs.

## Run
```bash

# install Helper libraries, for ubuntu 16.04 only
sudo apt-get install python3-pip python3-tk
pip3 install tensorflow numpy Pillow scipy opencv-python --user

# clone the repo
git clone git@github.com:zxdong262/tf-captcha-reader.git
cd tf-captcha-reader
npm i
python3 main.py

```




