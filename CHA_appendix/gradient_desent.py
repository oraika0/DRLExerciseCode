# import numpy as np
# def f(x):
#     return np.log(np.power(x,4) + np.power(x,3) + 2)
# def dfdx(x):
#     return (4 * np.power(x,3) + 3 * np.power(x,2)) / f(x)
# x = -9.41
# lr = 0.001
# epoch = 5000

# for i in range(epoch):
#     deriv = dfdx(x)
#     x = x - lr * deriv
 

# print(x)


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(np.power(x, 4) + np.power(x, 3) + 2)

def dfdx(x):
    return (4 * np.power(x, 3) + 3 * np.power(x, 2)) / f(x)

x_initial = -9.41
lr = 0.01
epoch = 5000

x_values = []
y_values = []

x = x_initial
for i in range(epoch):
    x_values.append(x) 
    y_values.append(f(x))
    deriv = dfdx(x)
    x = x - lr * deriv

# Plotting
x_range = np.linspace(-10, 10,800)
print(x_range)
y_range = f(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='Function $f(x)$')
plt.scatter(x_values, y_values, color='red', label='Gradient Descent Path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
