import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from keras.models import load_model

model = load_model("./MobileNet.h5")

class face_rec():
    def __init__(self):
        # 创建mtcnn对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5,0.8,0.9]

    def recognize(self,draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)
        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))

            new_img,_ = utils.Alignment_1(crop_img,landmark)

            new_img = (new_img.reshape(1, 160, 160, 3))
            new_img = np.array(new_img) / 255.0

            pred = model.predict(new_img)
            face_encoding = np.argmax(pred, axis=1)
            face_encodings.append(face_encoding)

        rectangles = rectangles[:,0:4]
        glasses_list = ('Glasses','No_glasses')

        face_names = []
        for i in face_encodings:
            glasses_list = glasses_list[int(face_encoding)]
            face_names.append(glasses_list)

        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles,face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw

if __name__ == "__main__":

    dududu = face_rec()

    #识别图片
    img = cv2.imread('./test/4.jpg')
    dududu.recognize(img)
    img = cv2.resize(img,(1024,512))
    cv2.imshow('Video', img)
    cv2.waitKey(0)

    #识别视频
    # video_capture = cv2.VideoCapture(0)
    # while True:
    #     ret, draw = video_capture.read()
    #     dududu.recognize(draw)
    #     cv2.imshow('Video', draw)
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break
    #
    # video_capture.release()
    # cv2.destroyAllWindows()