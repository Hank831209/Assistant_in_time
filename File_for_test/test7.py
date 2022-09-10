import numpy as np

B = np.zeros((2, 3, 3))
print(B)
print(B.shape)
print('---'*30)
A = np.full_like(B, 255)
print(A)
print(A.shape)

