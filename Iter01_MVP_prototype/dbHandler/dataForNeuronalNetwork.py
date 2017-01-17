import numpy as np
import SimpleITK as sitk
import os
import image_handler as ih


def data_for_neuronal_network(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path):

    ih.createPathIfNotExists(vol_dest_path)
    ih.createPathIfNotExists(seg_dest_path)

    vol_list = os.listdir(vol_src_path)  # vol_list: list of all files in volume directory

    print ("List of input files:")
    print (vol_list)

    iteration_num = 0

    for f in vol_list:
        print("iteration: " + str(iteration_num))
        print("read file...", f)

        #read CT volume files

        tmp_img = sitk.ReadImage(vol_src_path + "\\" + f)
        input_vol = sitk.GetArrayFromImage(tmp_img) # convert from image to matrix

        # read left bones segmentation

        print("read left bones segmentation...")
        tmp = sitk.ReadImage(seg_src_path + "\\" + ih.getSegFileName(seg_src_path,f,ih.bonesLId))
        left_bones_seg = sitk.GetArrayFromImage(tmp)  # convert from image to matrix


        # read right bones segmentation

        print('read right bones segmentation...')
        tmp = sitk.ReadImage(seg_src_path + "\\" + ih.getSegFileName(seg_src_path, f, ih.bonesRId))
        right_bones_seg = sitk.GetArrayFromImage(tmp) # convert from image to matrix


        # make one bones segmentation file

        print('merge left and right bones segmentation...')
        bones_seg = np.add(right_bones_seg, left_bones_seg)

        # turn segmentation matrices into 1.0 or 0.0 values
        bones_seg[np.where(bones_seg > 1)] = 1

        # image manifulations?
        # slice the image to fetches, and remove irrelevent pices
        iteration_num += 1

#run as:
vol_src_path = ""
seg_src_path = ""
vol_dest_path = ""
seg_dest_path = ""
#data_for_neuronal_network(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path)