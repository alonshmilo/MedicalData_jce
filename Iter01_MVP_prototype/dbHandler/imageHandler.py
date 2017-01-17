import numpy as np
import SimpleITK as stk
import os
from PIL import Image
import random
from random import shuffle
import tensorflow as tf


#classification index value

nothing = 0
bone = 1
bonesLId = 8
bonesRId = 9

# this function, checkes if path exisits, and if isn't exisits create this path.

def createPathIfNotExists(path):
    if not (os.path.exists(path)):
        os.makedirs(path)

# seg_list - all the files and directories in path,
# sub_name - find the sub name of file_name and appent to this sub name "_index"
# if this function find some sub-name,in seg_list, return it.
# return the file that contain the sub_name, otherwise return false

def getSegFileName(path, file_name, index):
    seg_list = os.listdir(path)

    sub_name = file_name[0:len(file_name) - 7] + "_" + str(index)
    for f in seg_list:
        if (f.startswith(sub_name)):
            return f
    return "Fail"

