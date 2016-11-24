import os
for root, dirs, files in os.walk("/mydir"):
    for file in files:
        if file.endswith(".jpg"):
             print(os.path.join(root, file))


'''
from os import walk
import os
mypath = os.path.dirname(os.path.abspath(__file__))
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
print mypath
print f
'''
