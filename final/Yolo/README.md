![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Support
In order to use Yolo of Darkenet for our project:
- Clone this folder
- Press "make" on ternminal
- Use the right config files using the instructions on darknet website: cfg/bone.cfg, cfg/bone.data, cfg/bone.names
- Use the right weights file (currenty - 1/yolo-bone_10000.weights), download it from [here](https://drive.google.com/open?id=0B4QF5OAaMsMNckpCUmpEa1FpRTQ).
- Detecting USAGE is: ./darknet detect [cfg file] [weights file] [image]
- Better USAGE is: ./darknet detector test [data file] [cfg file] [weights file] [image] 
- Training USAGE is: ./darknet detector train [data file] [cfg file] [weights file]

# Darknet
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


