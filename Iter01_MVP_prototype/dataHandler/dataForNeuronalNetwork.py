import os
import numpy as np
import nibabel as nib
import imageHandler as ih

def data_for_neuronal_network(seg_src_path, seg_dest_path):

    ih.createPathIfNotExists(seg_dest_path)

    seg_list = os.listdir(seg_src_path) # seg_list: list of all files in segmentation directory

    print ("List of input files:")
    print (seg_list)

    file_num = 0

    for f in seg_list:
        print("file number: " + str(file_num))
        print("read file...", f)

        path = seg_src_path + "/" + f
        print(path)

        image = nib.load(path)
        headers = image.header
        # print(headers)

        matrix = image.affine
       # matrix[np.where(matrix > 1)] = 1
       # matrix[np.where(matrix < 1)] = 0
        print(matrix)

        # manipulations.....
        #numbers

        dest_file = seg_dest_path + "/" + f
        nib.save(image, dest_file)


#run as:
seg_src_path = "/Users/stavbarazani/Desktop/TrainingData"
seg_dest_path = "/Users/stavbarazani/Desktop/result"
data_for_neuronal_network(seg_src_path, seg_dest_path)