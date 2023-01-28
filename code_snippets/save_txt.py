import numpy as np
import os

curr_dir = os.getcwd() + '/'
# Dummy NDP array of size (6, 3)
ndp_array = np.array([[-33989, -27089, -25778],
                      [-34021, -26358, -25645],
                      [-33988, -27166, -25088],
                      [-26293, -27623, -31753],
                      [-26809, -34008, -26540],
                      [-26448, -34023, -26294]])

# Note: You must set delimiter to '\t' and newline to '\n'. Otherwise, points will be deducted.
np.savetxt(curr_dir + "sample.txt", 
            ndp_array,
            delimiter='\t',
            newline='\n')