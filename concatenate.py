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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import scipy.misc
from matplotlib.pyplot import imread, imsave

import numpy as np


def main():
    data_son = sorted(glob.glob("./datasets/test/input/*.png"))
    data_rea = sorted(glob.glob("./datasets/test/gt/*.png"))
    data_fak = sorted(glob.glob("./results/*.png"))

    assert(len(data_son) == len(data_rea) == len(data_fak))
    for i in range(len(data_fak)):
        image_son = imread(data_son[i]).astype(np.float)
        image_son = scipy.misc.imresize(image_son, [256, 256])

        image_rea = imread(data_rea[i]).astype(np.float)
        image_rea = scipy.misc.imresize(image_rea, [256, 256])

        image_fak = imread(data_fak[i]).astype(np.float)
        image_fak = scipy.misc.imresize(image_fak, [256, 256])

        image_res = np.concatenate((image_son[:,:,:3], image_fak[:,:,:3], image_rea[:,:,:3]), axis=1)
        imsave("./presentation/test_{:04d}.png".format(i), image_res)
        print("CONCATENATING: Finished Image {:04d}".format(i))


if __name__ == '__main__':
    main()
