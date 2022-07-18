import numpy as np

a = np.array([[1574.0, 1816.0], [2435.0, 1821.0]])
one = np.array([[1], [1]])
one2 = np.array([[1, 1]])
print(np.hstack((a, one)))