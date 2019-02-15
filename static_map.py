# Copyright 2018 Giovanni Giacomo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob as glob
import numpy as np

from matplotlib.pyplot import imread, imsave

def main():
    data = sorted(glob.glob("./datasets/old/train/gt/*.png"))

    for index in range(len(data)):
        img = imread(data[index]).astype(np.float)
        new_img = np.zeros(shape=(img.shape[0], img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]

                if pixel[0] == 1.0:
                    new_img[i, j] = 0.0
                elif pixel[1] == 1.0:
                    new_img[i, j] = 1.0
                elif pixel[2] == 1.0:
                    new_img[i, j] = 0.0
        imsave(data[index].replace("/old", "", 1), new_img)

if __name__ == '__main__':
    main()
