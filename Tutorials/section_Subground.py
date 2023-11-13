import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Subground_HVSR import *

# File with data here
# Requires formatting to the output files of HVSR vs depth from the 
# routine "virtual_Borehole". These were named (self.filename)+"_depth_"+str(chainage)+"m.dat".
# This routine looks for a list of them, eg. blah_depth_0m.dat, blah_depth_10m.dat

file_name = "HVSR"


# This all comes from the header of the data file
# The tromino file format changes the header size and position
# Best just to manually input these to avoid tears. 
# Note can define column order, here it defaults to N, E, Z
#freq = 512
#time1 = "09:24:39"
#time2 = "09:46:39"
#header_lines = 33

# Note the example data has not been normalised, so we send a flag to say say (0), default is normalised (=1) 
flow1 = HVSR_Processing(filename = file_name,normalised=0)

flow1.plot_HVSR_section()
plt.close()
img = mpimg.imread("HVSR_gridded2.png")
plt.axis('off')
plt.imshow(img)
plt.show()





