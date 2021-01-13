import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

plt.plot([1, 2, 3, 4, 5], [1, 0, 0, 0, 0])
plt.plot([1, 2, 3, 4, 5], [0, 1, 0, 0, 0])
plt.plot([1, 2, 3, 4, 5], [0, 0, 1, 0, 0])
plt.plot([1, 2, 3, 4, 5], [0, 0, 0, 1, 0])
plt.plot([1, 2, 3, 4, 5], [0, 0, 0, 0, 1])
plt.show()
