import pandas as pd
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
#from wide_resnet import WideResNet
from TYY_model import TYY_2stream, TYY_1stream, TYY_MobileNet, TYY_MobileNet_2stream
from utils import mk_dir, load_data
from TYY_utils import load_data_npz
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
from keras.applications.mobilenet import MobileNet

logging.basicConfig(level=logging.DEBUG)


class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1*0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02*0.1
        elif epoch_idx < self.epochs * 0.75:
            return 0.004*0.1
        return 0.0008*0.1


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--evaluate", "-e", type=str, required=True,
                        help="path to evaluate database mat file")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
						help="use data augmentation if set true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input
    evaluate_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs

    validation_split = args.validation_split
    use_augmentation = args.aug

    logging.debug("Loading data...")
    #image, gender, age, _, image_size, _ = load_data(input_path)
    image, gender, age, _, image_size, _ = load_data_npz(input_path)
    
    x_data = image
    y_data_g = np_utils.to_categorical(gender, 2)

    #Quantize the age into 21 bins:

    #Remember that np.digitize take index for the quantization, 2 inds only have 1 bins, 3 inds have 2 bins, and so on.
    age_bins = np.linspace(0,100,21+1)#(Important!!!) Remember to add 1 to turn bins number to index number
    age_step = np.digitize(age,age_bins)-1#(Important!!!) Since np.digitize output start from 1, we want it start from zero, so we substract 1.
    y_data_a = np_utils.to_categorical(age_step, 21)
    
    alpha = 0.25
    model = TYY_MobileNet(image_size,alpha)()
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir("models")

    #with open(os.path.join("models", "WRN_{}_{}.json".format(depth, k)), "w") as f:
    with open(os.path.join("models", 'mobilenet_%s_%d_tf_no_top.json' % (alpha, image_size)), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints")
    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs)),
                 ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

    logging.debug("Running training...")
    


    data_num = len(x_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    x_data = x_data[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    
    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    if use_augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255))
        training_generator = MixupGenerator(x_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2,
                                            datagen=datagen)()
        hist = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(x_test, [y_test_g, y_test_a]),
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)
    else:
        hist = model.fit(x_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
						validation_data=(x_test, [y_test_g, y_test_a]))


    logging.debug("Saving weights...")
    #model.save_weights(os.path.join("models", "WRN_{}_{}.h5".format(depth, k)), overwrite=True)
    model.save_weights(os.path.join("models", 'mobilenet_%s_%d_tf_no_top.h5' % (alpha, image_size)), overwrite=True)
    #pd.DataFrame(hist.history).to_hdf(os.path.join("models", "history_{}_{}.h5".format(depth, k)), "history")
    pd.DataFrame(hist.history).to_hdf(os.path.join("models", 'history_mobilenet_%s_%d_tf_no_top.h5' % (alpha, image_size)), "history")


    #Evaluation
    image, gender, age, _, image_size, _ = load_data(evaluate_path)
    x_evaluate = image
    y_evaluate_g = np_utils.to_categorical(gender, 2)
    age_bins = np.linspace(0,100,21+1)#(Important!!!) Remember to add 1 to turn bins number to index number
    age_step = np.digitize(age,age_bins)-1#(Important!!!) Since np.digitize output start from 1, we want it start from zero, so we substract 1.
    y_evaluate_a = np_utils.to_categorical(age_step, 21)

    metrics = model.evaluate(x_evaluate,[y_evaluate_g,y_evaluate_a],batch_size=batch_size,verbose=1)
    for i in range(len(model.metrics_names)):
    	print(model.metrics_names[i]+": "+metrics[i].astype(str))

if __name__ == '__main__':
    main()