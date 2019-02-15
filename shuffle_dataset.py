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
import os
import random
import scipy.misc

import numpy as np

def main():
    data_son = sorted(glob.glob("./datasets/input/*.png"))
    data_rea = sorted(glob.glob("./datasets/gt/*.png"))

    assert (len(data_son) == len(data_rea))

    indices = list(range(0, len(data_son)))
    for i in range(len(data_son)):
        new_index = random.choice(indices)
        os.rename(data_son[i], "./datasets/input/train_%05d.png" % new_index)
        os.rename(data_rea[i], "./datasets/gt/real_%05d.png" % new_index)
        indices.remove(new_index)


if __name__ == '__main__':
    main()
