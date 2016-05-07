import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('velocity.csv',delimiter=',', dtype = float)

time = [row[0] for row in data]
velocity = [row[1] for row in data]

axes = plt.plot(time, velocity)
plt.ylabel("Velocity (km/h)")
plt.xlabel("Time (s)")
plt.title("JCSAT-14")
plt.show()