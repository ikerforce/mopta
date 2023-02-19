import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

pd = truncnorm((20 - 100) / 50, (250 - 100) / 50, loc=100, scale=50)
samples = pd.rvs(size=10790)
plt.hist(samples, bins=25)
plt.show()