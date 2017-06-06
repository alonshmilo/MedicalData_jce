from PIL import Image
import sys


import glob, os
x = 0000
for file in glob.glob("*.png"):
    x = x+1
    im = Image.open(file)
    im.save("new_" + str(x)+"object.jpg", "JPEG")
    os.remove(file)
    print(file)

for file in glob.glob("*.jpeg"):
    x = x+1
    im = Image.open(file)
    im.save("axial"+str(x)+".JPEG", "JPEG")
    os.remove(file)
    print(file)
