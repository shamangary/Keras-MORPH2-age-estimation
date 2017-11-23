# Keras-MORPH-age-estimation
Keras implementation for MORPH dataset



## How to run?
+ Step.1
```
python TYY_MORPH_create_db.py --output morph_db.npz
```

+ Step.2
```
KERAS_BACKEND=tensorflow python TYY_train_MORPH.py --input ./data/morph_db.npz
```


