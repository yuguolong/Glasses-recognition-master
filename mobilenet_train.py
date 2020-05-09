from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.layers import SeparableConv2D,Dropout,Dense,MaxPooling2D,Flatten,GlobalAveragePooling2D,Input
from keras.preprocessing.image import ImageDataGenerator

#model
base_model = MobileNet(alpha=0.25, depth_multiplier=1, dropout=1e-4,weights='imagenet', include_top=False)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64,activation='relu')(x)
x = Dropout(0.1)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#数据读入
train_datagen = ImageDataGenerator(rescale=1./255,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   # rotation_range=2,
                                   # zoom_range=0.3,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2
                                   )

# 测试集不许动，去均值中心化完了之后不许动
validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = './train_data/train'
validation_dir = './train_data/test'
# 利用python生成器不断的生成训练样本
train_generator = train_datagen.flow_from_directory(
    train_dir,
    # 缩放到356*356
    target_size=(160, 160),
    # 每个批量包含batch_size个样本
    batch_size=100,
    # 因为是单标签，多分类问题，最后损失函数要用catagorical_crossentropy,所以此处用catagorical
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(160, 160),
    batch_size=14,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    class_weight= [1,1.1],
    steps_per_epoch=8,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=3,
    verbose=1)

model.save('./MobileNet.h5')