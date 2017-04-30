## DeepMedic 

#### 1.2. Installation

The software can be found at `https://github.com/Kamnitsask/deepmedic/`. After cloning the project, all the dependencies can be installed by running the following command in the root directory:

```python
python setup.py install
```

This will download all required libraries, install and add them to the environment's PATH. This should be enough to use the DeepMedic.


### 2. Running the Software

#### 2.1 Train

We will here train a CNN model. This is for tests and real data. 

NOTE: First see [Section 1.2](#12-installation) for installation of the required packages. 
**Please remember to change the path if needed!!!**

Lets **create** a model :
```cshell
./deepMedicRun -newModel ./examples/configFiles/tinyCnn/model/modelConfig.cfg
```

This command parses the given configuration file, creates a CNN model with the specified architecture, initializes and saves it. The folder `./examples/output/` should have been created by the process, where all output is saved. When the process finishes (roughly after a couple of minutes) a new and untrained model should be saved using [cPickle](https://docs.python.org/2/library/pickle.html) at `./examples/output/cnnModels/tinyCnn.initial.DATE+TIME.save`. All output of the process is logged for later reference. This should be found at `examples/output/logs/tinyCnn.txt`. Please make sure that the process finishes normally, the model and the logs created. If everything looks fine, briefly rejoice and continue... 

Lets **train** the model with the command (replace *DATE+TIME*):
```cshell
./deepMedicRun -train examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg \
                       -model examples/output/cnnModels/tinyCnn.initial.DATE+TIME.save
```

The model should be loaded and training for two epochs should be performed. After each epoch the trained model is saved at `examples/output/cnnModels/trainSessionWithValidTinyCnn`. The logs for the sessions should be found in `examples/output/logs/trainSessionWithValidTinyCnn.txt`. Finally, after each epoch, the model performs segmentation of the validation images and the segmentation results (.nii files) should appear in `examples/output/predictions/trainSessionWithValidTinyCnn/predictions/`. If the training finishes normally (should take 5 mins) and you can see the mentioned files in the corresponding folders, beautiful. 

You can **plot the training progress** using an accompanying script, which parses the training logs:
'''
python plotTrainingProgress.py examples/output/logs/trainSessionWithValidTinyCnn.txt -d
'''

Now lets **test** with the trained model (replace *DATE+TIME*):
```cshell
./deepMedicRun -test examples/configFiles/tinyCnn/test/testConfig.cfg \
                       -model examples/output/cnnModels/trainSessionWithValidTinyCnn/tinyCnn.trainSessionWithValidTinyCnn.final.DATE+TIME.save
```

This should perform segmentation of the testing images and the results should appear in `examples/output/predictions/testSessionTinyCnn/` in the `output` folder. In the `features` folder you should also find some files, which are feature maps from the second layer. DeepMedic gives you this functionality (see testConfig.cfg). If the testing process finishes normally and all output files seem to be there, **everything seems to be working!**