# Deep Medic

Table of Contents
=================

* [1. Installation and Requirements](http://)  
  * [1.1 Required Libraries](http://)    
  * [1.2 Installation](http://)  
  * [1.3 Required Data Pre-Processing](http://)   
  * [1.4  GPU Processing](http://)  

## 1. Installation and Requirements

**1.1. Required Libraries**

The system is written in python. The following libraries are required:

  * [Theano](http://deeplearning.net/software/theano/): This is the Deep Learning library.
  * [Nose](https://pypi.python.org/pypi/nose/): Required for Theano’s unit tests.
  * [NiBabel](http://nipy.org/nibabel/): The library used for loading NIFTI files.
  * [Parallel Python](http://www.parallelpython.com/): Library used to parallelize parts of the training process.
  * [six](https://pypi.python.org/pypi/six): Python compatibility library.
  * [scipy](https://www.scipy.org/): Package of tools for statistics, optimization, integration, algebra, machine learning.
  * [numpy](http://www.numpy.org/): General purpose array-processing package.
  
  **1.2. Installation**
  
  The software can be found at final/deepmedic/. 
  After cloning the project, all the dependencies can  be installed by running the following command in the root directory:
  
     python setup.py install
     
  This will download all required libraries, install and add them to the environment's PATH.
  This should be enough to use the DeepMedic.
  
  Alternatively, the user can manually add them to the PATH, or use the provided environment.txt file:
  
       #=============== LIBRARIES ====================
       #Theano is the main deep-learning library used. Version >= 0.8 required. Link: http://deeplearning.net/software/theano/
       path_to_theano = '/path/to/theano/on/the/filesystem/Theano/'

       #Nose is needed by Theano for its unit tests. Link: https://pypi.python.org/pypi/nose/
       path_to_nose = '/path/to/nose/on/the/filesystem/nose_installation'

      #NiBabel is used for loading/saving NIFTIs. Link: http://nipy.org/nibabel/
       path_to_nibabel = '/path/to/nibabel/on/the/filesystem/nibabel'

       #Parallel-Python is required, as we extract training samples in parallel with gpu training. 
       Link:     http://www.parallelpython.com/
       path_to_parallelPython = '/path/to/pp/on/the/filesystem/ppBuild'

The latter file is parsed by the main software.
If the lines with the corresponding lines are not commented out, the given path will be internally pre-pended in the PATH.

**1.3 Required Data Pre-Processing**

* DeepMedic processes **NIFTI files only**. All data should be in the .nii format.
* The input modalities, ground-truth labels, ROI masks and other images of each subject need to be co-registered 
 (per -subject, no need for inter-subject registration).
* The images of each subject should **have the same dimensions** (per subject, no need for whole database).
 This is, the number of voxels per dimension must be the same for all images of a subject.
* **Resample all images in the database to the same voxel size**. The latter is needed because the kernels (filters) of the DeepMedic need to correspond to the same real-size patterns (structures) **for all subjects**.
* Make sure that the **ground-truth labels** for training and evaluation represent the background with zero.
 The system also assumes that the task’s classes are indexed increasing by one (not 0,10,20 but 0,1,2).
* You are strongly advised to normalize the intensity of the data within the ROI to a zero-mean, unary-variance space. Our default configuration significantly underperforms if intensities are in another range of values.

**1.4. GPU Processing**
Small networks can be run on the cpu.
But 3D CNNs of considerable size require processing on the GPU.
For this, an installation of [Nvidia’s CUDA](https://developer.nvidia.com/cuda-toolkit) is needed.
Make sure to acquire a version compatible with your GPU drivers.
Theano needs to be able to find CUDA’s compiler, the nvcc, in the environment’s path.
It also dynamically links to cublas.so libraries, which need to be visible in the environment’s.
Prior to running DeepMedic on the GPU, you must manually add the paths to the folders containing these files in your environment's variables.
As an example, in a cshell this can be done with setenv:

    setenv PATH '/path-to/cuda/7.0.28/bin':$PATH
    setenv LD_LIBRARY_PATH '/path-to/cuda/7.0.28/lib64
    




