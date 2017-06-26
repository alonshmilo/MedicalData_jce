# Deep Medic

Table of Contents
=================

* [1. Installation and Requirements](http://)  
  * [1.1 Required Libraries](http://)    
  * [1.2 Installation](http://)  
  * [1.3 Required Data Pre-Processing](http://)   
  * [1.4  GPU Processing](http://)  
* [2. Running the Software](http://)  

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
    
**2. Running the Software**    
In the ribDeepMedic/configFiles folder we provide two sets of configuration files.
Firstly, the configuration of a very small network is given in ribDeepMedic/configFiles/tinyCnn/.
This network can be trained within minutes on a CPU. 
We also provide the full configuration of the DeepMedic model in the folder ribDeepMedic/configFiles/deepMedic/.

**TO create a model** :

     ./deepMedicRun -newModel ./examples/configFiles/tinyCnn/model/modelConfig.cfg
This command parses the given configuration file, creates a CNN model with the specified architecture, initializes and saves it.
The folder ./ribDeepMedic/output/ should have been created by the process, where all output is saved.
When the process finishes (roughly after a couple of minutes) a new and untrained model should be saved at 
                 ./ribDeepMedic/output/cnnModels/tinyCnn.initial.DATE+TIME.save.
All output of the process is logged for later reference.
This should be found at:
                 ribDeepMedic/output/logs/tinyCnn.txt.
Please make sure that the process finishes normally, the model and the logs created.
If everything looks fine, briefly rejoice and continue...

**To train the model with the command (replace DATE+TIME):**

     ./deepMedicRun -train ribDeepMedic/configFiles/tinyCnn/train/trainConfigWithValidation.cfg \
     -model ribDeepMedic/output/cnnModels/tinyCnn.initial.DATE+TIME.save

The model should be loaded and training for two epochs should be performed (in tinyCnn).
After each epoch the trained model is saved at ribDeepMedic/output/cnnModels/trainSessionWithValidTinyCnn.
The logs for the sessions should be found in ribDeepMedic/output/logs/trainSessionWithValidTinyCnn.txt.
Finally, after each epoch, the model performs segmentation of the validation images and the segmentation results (.nii files) should appear in ribDeepMedic/output/predictions/trainSessionWithValidTinyCnn/predictions/.
If the training finishes normally (should take 5 mins) and you can see the mentioned files in the corresponding folders, beautiful.

**To test with the trained model (replace DATE+TIME):**

     ./deepMedicRun -test ribDeepMedic/configFiles/tinyCnn/test/testConfig.cfg \
     -model ribDeepMedic/output/cnnModels/trainSessionWithValidTinyCnn/ 
     tinyCnn.trainSessionWithValidTinyCnn.final.DATE+TIME.save

This should perform segmentation of the testing images and the results should appear in ribDeepMedic/output/predictions/testSessionTinyCnn/ in the output folder.
If the testing process finishes normally and all output files seem to be there, everything seems to be working! On the CPU...

If using the DeepMedic on the GPU is alright on your system.
First, delete the ribDeepMedic/output/ folder for a clean start.
Now, most importantly, place the path to CUDA's nvcc into your PATH and to the cublas.so in your LD_LIBRARY_PATH 
(see section 1.4)

You need to perform the steps we did before for creating a model, training it and testing with it, but on the GPU. To do this, repeat the previous commands and pass the additional option -dev gpu.
For example:

     ./deepMedicRun -dev gpu -newModel ./examples/configFiles/tinyCnn/model/modelConfig.cfg

**Note:** The config files are parsed as python scripts, thus follow python syntax. Any commented-out configuration variables are internally given default values.

                 [-h] [-newModel MODEL_CONF] [-train TRAINING_CONF]
                 [-test TESTING_CONF] [-model SAVED_MODEL] [-dev DEVICE]
                 [-pretrained PRETRAINED_MODEL]
                 [-layers LAYERS_TO_TRANSFER [LAYERS_TO_TRANSFER ...]]
                 [-resetOptimizer]

 **-h, --help** show this help message and exit.

 **-newModel MODEL_CONF** Create a new CNN model with model parameters at given config file [MODEL_CONF].

 **-train TRAINING_CONF**  Train a CNN model with training parameters at given config file [TRAINING_CONF].
 This option can follow a [-newModel MODEL_CONF] option, to create a new model and start training it immediately.
 Otherwise, existing model can be specified in the training-config file or by the additional option [-model].

 **-test TESTING_CONF** Test with an existing CNN model. The testing session's parameters should be given in 
 config file [TESTING_CONF].
 Existing model can be specified in the given training-config file or by the additional option [-model].
 This option cannot be used in combination with [-newModel] or [-train].

 **-model SAVED_MODEL** The path to a saved existing cnn model, to train or test with.
  This option can follow a [-train] or [-test] option. Not in combination with a [-newModel].
  If given, this option will overide any "model" parameters given in the train or testing configuration file.

 **-dev DEVICE** Specify the device to run the process on. Values: [cpu] or [gpu] (default = cpu).
 In the case of multiple GPUs, specify a particular GPU device with a number, in the format: gpu2.
 NOTE: For GPU processing, CUDA libraries must be first added in your environment's PATH and LD_LIBRARY_PATH.

 **-pretrained PRETRAINED_MODEL** Use to transfer the weights from a previously trained model to a new model.
 This option must follow a [-newModel] option.
 Usage: ./deepMedicRun -newModel /path/to/model/config -pretrained /path/to/pretrained/model 
 NOTE: By default, parameters are transfered to all layers except the classification layer. Use option [-layers] 
 to manually specify layers to pretrain.

 **-layers LAYERS_TO_TRANSFER [LAYERS_TO_TRANSFER ...]** Use only after a [-pretrained] option.
 Specify to which layers of the new model parameters should be transferred to.
 First layer is 1. Classification layer of original DeepMedic is 11. Same layers from each parallel-pathway will 
 be transfered.
 Usage: ./deepMedicRun -newModel /path/to/model/config -pretrained /path/to/pretrained/model -layers 1 2 3 ...

 **-resetOptimizer** Use optionally with a [-train] command. Does not take an argument.
 Usage: ./deepMedicRun -train /path/to/train/config -resetOptimizer ...etc...
 Resets the model's optimization state before starting the training session (eg number of epochs already trained, 
 current learning rate etc).
 IMPORTANT: Trainable parameters are NOT reinitialized! 
 Useful to begin a secondary training session with new learning-rate schedule, in order to fine-tune a previously 
 trained model (Doc., Sec. 3.2)
