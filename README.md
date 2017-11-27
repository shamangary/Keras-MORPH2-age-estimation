# Keras-MORPH-age-estimation
Keras implementation for MORPH dataset age estimation.

This project contains **Mobilenet and Densenet** with **regression and DEX framework**.

## Update(2017/11/27)
+ Change training default epoch to 90
+ Decay learning rate at epoch [30,60]

## How to run?
+ Step.1
Download MORPH dataset
https://www.faceaginggroup.com/morph/ **Unzip it under './MORPH'**

You have to apply for the dataset. No easy way to download it unfortunately :(


+ Step.2 Preprocess the dataset **(change isPlot inside TYY_MORPH_create_db.py to True if you want to see the process)**
```
python TYY_MORPH_create_db.py --output morph_db.npz
```

+ Step.3 Run the training and evalutation **(change netType inside TYY_train_MORPH.py for different networks)**
```
KERAS_BACKEND=tensorflow python TYY_train_MORPH.py --input ./morph_db.npz
```

## Training and evaluation

+ Training ratio: 0.8
+ Validation ratio: 0.2

+ Evaluation metric:
Mean-absoluate-error (MAE) -> name: **val_pred_a_mean_absolute_error**

+ Output example:
```
pred_a_softmax_loss: 2.4073 - pred_a_loss: 9.4221 - pred_a_softmax_acc: 0.1183 - pred_a_mean_absolute_error: 9.4221 - val_loss: 2.4423 - val_pred_a_softmax_loss: 2.4423 - val_pred_a_loss: 9.4864 - val_pred_a_softmax_acc: 0.1339 - val_pred_a_mean_absolute_error: 9.4864
```
## Parameters

+ DEX: num_neu is the output dimension of the classfication training part. Range of num_neu: [1~101]
+ Mobilenet: alpha is the paramters to control the network size. Recommended value of alpha: 1, 0.5, 0.25
+ Densenet: densenet_depth is the depth of the network (Obviously~~) 

## Dependencies
+ Keras
+ Tensorflow
+ anaconda
+ python3
+ opencv3
+ dlib
+ moviepy
+ pytables


## References
+ https://github.com/yu4u/age-gender-estimation
+ https://github.com/titu1994/DenseNet
+ R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image without facial landmarks," IJCV, 2016.
+ https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py
