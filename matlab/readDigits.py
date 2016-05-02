import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb

pach = np.zeros((16, 240))

for i in range(0,10):
	name = 'digit' + str(i) + '.bin'

	digit = np.fromfile(name, dtype=np.uint8, count=-1).reshape((28, 28),  order='F')

	plt.imshow(digit)
	plt.show()

	#pdb.set_trace()
