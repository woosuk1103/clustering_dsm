import numpy as np

total = []

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[11,22,33],[44,55,66],[77,88,99]])

total.append(a)
print(total, '1st')

total.append(b)
print(total, '2nd')

print(total[0].shape, '3rd')
a = np.reshape(a, (-1,1))

print(a)
print(a.shape)

a = np.squeeze(a)

print(a)
print(a.shape)

print(3==3.0)
