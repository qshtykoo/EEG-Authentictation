import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


raw_psec = pd.read_csv('testsubject1/2018-03-17/1.powerspec.csv')
header_list = list(raw_psec)

index_list = np.array(range(122))
header_list_filtered = [ header_list[index] for index in index_list ] #121 frequency bins

header_list = header_list[1:]
header_list_filtered = header_list_filtered[1:]

header_list_float = list(map(float, header_list))
header_list_filtered = list(map(float, header_list_filtered))

first_second = raw_psec.loc[0:7, :]
selected_time = np.sum(raw_psec.iloc[0:35, 1:122], axis =0)
plt.plot(header_list_filtered, selected_time)
plt.show()
