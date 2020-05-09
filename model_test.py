from keras.models import load_model
import numpy as np
import cv2#加载模型h5文件
model = load_model("./MobileNet.h5")
model.summary()

input = cv2.imread('440.jpg')
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
input = (input.reshape(1,160,160,3))
pre_x = np.array(input) / 255.0

predict = model.predict(pre_x)
predict = np.argmax(predict,axis=1)
print(predict)
print(type(predict))
b = ['glasses','no_glasses']
print(b[int(predict)])
