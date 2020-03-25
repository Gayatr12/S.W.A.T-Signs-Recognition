import os
import glob

BASE_PATH = '/Users/RayM/Documents/School/CompVision/Project3/new_test_data'
VID_PATH = os.path.join(BASE_PATH, '**', '**.npy') 

filelist= glob.glob(VID_PATH)


# Write to file
with open('new_paths.txt', 'w') as filehandle:
    for listitem in filelist:
        filehandle.write('%s\n' % listitem)

