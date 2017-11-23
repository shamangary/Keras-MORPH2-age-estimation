import pandas as pd
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from TYY_model import TYY_MobileNet_reg, TYY_MobileNet_dex, TYY_DenseNet_reg, TYY_DenseNet_dex
from TYY_utils import mk_dir, load_MORPH_data_npz
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet

logging.basicConfig(level=logging.DEBUG)




def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split


    logging.debug("Loading data...")
    image, gender, age, image_size = load_MORPH_data_npz(input_path)
    
    x_data = image
    y_data_a = age

    netType = 3

    if netType == 1:
        alpha = 1
        model = TYY_MobileNet_reg(image_size,alpha)()
        save_name = 'mobilenet_reg_%s_%d' % (alpha, image_size)
        model.compile(optimizer=Adam(), loss=["mae"], metrics={'pred_a':'mae'})

    elif netType == 2:
        N_densenet = 5
        depth_densenet = 3*N_densenet+4
        model = TYY_DenseNet_reg(image_size,depth_densenet)()
        save_name = 'densenet_reg_%d_%d' % (depth_densenet, image_size)
        model.compile(optimizer=Adam(), loss=["mae"], metrics={'pred_a':'mae'})

    elif netType == 3:
        num_neu = 25
        alpha = 0.25
        model = TYY_MobileNet_dex(image_size,alpha,num_neu)()
        save_name = 'mobilenet_dex_%s_%d_%d' % (alpha, image_size, num_neu)
        model.compile(optimizer=Adam(), loss=["categorical_crossentropy","mae"],loss_weights=[1,0], metrics={'pred_a_softmax':'accuracy','pred_a':'mae'})

    elif netType == 4:
        num_neu = 25
        N_densenet = 3
        depth_densenet = 3*N_densenet+4
        model = TYY_DenseNet_dex(image_size,depth_densenet,num_neu)()
        save_name = 'densenet_dex_%d_%d_%d' % (depth_densenet, image_size, num_neu)
        model.compile(optimizer=Adam(), loss=["categorical_crossentropy","mae"],loss_weights=[1,0], metrics={'pred_a_softmax':'accuracy','pred_a':'mae'})



    
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir("models")

    with open(os.path.join("models", save_name+'.json'), "w") as f:
        f.write(model.to_json())

    mk_dir("checkpoints")
    callbacks = [ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")]

    logging.debug("Running training...")
    


    data_num = len(x_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    x_data = x_data[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    
    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    if netType == 3 or netType == 4:
        age_bins = np.linspace(0,100,num_neu+1)#(Important!!!) Remember to add 1 to turn bins number into index number
        y_train_a_class = np.digitize(y_train_a,age_bins)-1#(Important!!!) Since np.digitize output start from 1, we want it start from zero, so we substract 1.
        y_train_a_class = np_utils.to_categorical(y_train_a_class, num_neu)
        y_test_a_class = np.digitize(y_test_a,age_bins)-1#(Important!!!) Since np.digitize output start from 1, we want it start from zero, so we substract 1.
        y_test_a_class = np_utils.to_categorical(y_test_a_class, num_neu)
        
        hist = model.fit(x_train, [y_train_a_class,y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks, validation_data=(x_test, [y_test_a_class,y_test_a]))
    else:
        hist = model.fit(x_train, [y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks, validation_data=(x_test, [y_test_a]))
    

    logging.debug("Saving weights...")
    model.save_weights(os.path.join("models", save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join("models", 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()