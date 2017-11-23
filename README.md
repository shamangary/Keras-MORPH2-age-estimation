# Keras-MORPH-age-estimation
Keras implementation for MORPH dataset
This project contains Mobilenet and Densenet with regression and DEX framework.


## How to run?
+ Step.1
Download MORPH dataset
https://www.faceaginggroup.com/morph/


+ Step.2 Preprocess the dataset
```
python TYY_MORPH_create_db.py --output morph_db.npz
```

+ Step.3 Run the training and testing
```
KERAS_BACKEND=tensorflow python TYY_train_MORPH.py --input ./data/morph_db.npz
```

## References
https://github.com/yu4u/age-gender-estimation
https://github.com/titu1994/DenseNet
