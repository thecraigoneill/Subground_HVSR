import numpy as np
import matplotlib.pyplot as plt
from Subground_HVSR import *

# File with data here
file_name = "example_tromino_data.dat"


# This all comes from the header of the data file
# The tromino file format changes the header size and position
# Best just to manually input these to avoid tears. 
# Note can define column order, here it defaults to N, E, Z
freq = 512
time1 = "09:24:39"
time2 = "09:46:39"
header_lines = 33


flow1 = HVSR_Processing(filename = file_name, freq=freq,time1=time1,time2=time2,header_lines = header_lines)

flow1.import_HVSR()
#flow1.plot_raw()

# Select windows from previous plot for filtering

freq_win = np.array( [ [100,190], [220,271],[300,415],[515,833], [871,1133],[1161,1178],[1216,1281] ])


flow1.freq_win = freq_win
flow1.filter_HVSR_time()
#flow1.plot_filtered()

# Create HVSR periodogram

flow1.HVSR_periodogram()
flow1.plot_HVSR()
flow1.virtual_Borehole()

img = plt.imread(str(flow1.filename)+"hvsr_borelog.png")
plt.axis('off')
plt.imshow(img)




