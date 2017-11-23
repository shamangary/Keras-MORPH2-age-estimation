import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import sys
import dlib
from moviepy.editor import *

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    img_size = args.img_size

    mypath = './MORPH'
    isPlot = False
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    detector = dlib.get_frontal_face_detector()

    out_genders = []
    out_ages = []
    out_imgs = []

    for i in tqdm(range(len(onlyfiles))):
        
        img_name = onlyfiles[i]
        temp_name = img_name.split('_')
        temp_name = temp_name[1].split('.')
        isMale = temp_name[0].find('M')
        isFemale = temp_name[0].find('F')
        
        if isMale > -1:
            gender = 0
            age = temp_name[0].split('M')
            age = age[1]
        elif isFemale > -1:
            gender = 1
            age = temp_name[0].split('F')
            age = age[1]
        
        out_genders.append(int(gender))
        out_ages.append(int(age))
        input_img = cv2.imread('./MORPH/'+img_name)
        img_h, img_w, _ = np.shape(input_img)
        detected = detector(input_img,1)
        if len(detected) == 0:
            img_clip = ImageClip(input_img)
            if isPlot:
                img_clip.show()
                key = cv2.waitKey(3000)
            input_img = cv2.resize(input_img, (img_size, img_size))
            out_imgs.append(input_img)
        elif len(detected) == 1:
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - 0.4 * w), 0)
                yw1 = max(int(y1 - 0.4 * h), 0)
                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 255, 0), 2)
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            
            if isPlot:
                img_clip = ImageClip(input_img)
                img_clip.show()
                key = cv2.waitKey(1000)

            out_imgs.append(faces[0,:,:,:])

    np.savez(output_path,image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages), img_size=img_size)

if __name__ == '__main__':
    main()
