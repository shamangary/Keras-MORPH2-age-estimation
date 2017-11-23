# Keras-MORPH-age-estimation
Keras implementation for MORPH dataset age estimation.

This project contains **Mobilenet and Densenet** with **regression and DEX framework**.


## How to run?
+ Step.1
Download MORPH dataset
https://www.faceaginggroup.com/morph/ **Unzip it under './MORPH'**


+ Step.2 Preprocess the dataset
```
python TYY_MORPH_create_db.py --output morph_db.npz
```

+ Step.3 Run the training and testing
```
KERAS_BACKEND=tensorflow python TYY_train_MORPH.py --input ./data/morph_db.npz
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


## Dependencies
+ Keras
+ Tensorflow
+ anaconda
+ python3
+ dlib
+ moviepy
+ pytables


## References
+ https://github.com/yu4u/age-gender-estimation
+ https://github.com/titu1994/DenseNet
+ R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image without facial landmarks," IJCV, 2016.
+ https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py
