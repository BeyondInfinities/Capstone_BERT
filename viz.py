import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import warnings
# fit an exponential distribution to the data:
from scipy.stats import expon
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_music = pd.read_csv("Results/music_variance.csv")
# plot the histrogram of the data
plt.hist(df_music['0'], bins=200)


# calculate the mean and standard deviation of the data
mean = np.mean(df_music['0'])
std = np.std(df_music['0'])
# plot the mean and standard deviation on the histogram
plt.axvline(mean, color='b', linestyle='dotted', linewidth=2, alpha=0.5)

# fit an exponential distribution to the data:
param = expon.fit(df_music['0'])

# now, param[0] and param[1] are the mean and 
# the standard deviation of the fitted distribution
x = np.linspace(0,1,100)
# fitted distribution
pdf_fitted = expon.pdf(x,loc=param[0],scale=param[1]) * 10
# original distribution
pdf = expon.pdf(x) 


plt.plot(x,pdf_fitted,'r-',x,pdf,'grey', alpha=0.5)
plt.title(f"Distribution of Genre (Music) Variance by attribute \n Mean: {mean:.2f} Std: {std:.2f}")
plt.xlabel('Variance')
plt.ylabel('Count')
plt.show()
