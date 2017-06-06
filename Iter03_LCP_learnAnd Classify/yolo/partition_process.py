import glob, os

"""This scripts is meant for partitioning our data to 2 parts: train and test.
    
    Usage: python partition_process.py
    
    To place this scripts inside the folder, aside with the images and txt files.
    Pay attention to change: path_data
    You can change percentage_test according to what you want.
    
    Output: 2 files: train.txt and test.txt in the same folder, holding the right lists."""

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the data will reside, relative to 'darknet.exe'
path_data = 'train/001/'

# Percentage of images to be used for the test set
percentage_test = 20;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)


for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.JPEG")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(path_data + title + '.jpg' + "\n")

    else:
        file_train.write(path_data + title + '.jpg' + "\n")
        counter = counter + 1
