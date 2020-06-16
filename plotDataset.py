import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
#%matplotlib qt


bank = pd.read_csv(r'data\bank-additional-full.csv', sep=';')


bank_X = bank.iloc[:,[i for i in range(0,len(bank.columns)-1)]]
bank_X = bank_X.drop('duration', axis=1) # giati to kanoume drop auto ?

target_names = ["yes", "no"]
all_columns = bank_X.columns
numerical_cols = bank_X._get_numeric_data().columns
categorial_columns = list(set(all_columns) - set(numerical_cols))

#labelencoder_X = LabelEncoder()
#for col in categorial_columns:
#    bank_X[col] = labelencoder_X.fit_transform(bank_X[col])
#    
#descr = bank_X.describe()



#Categorical parameters

fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(18, 12))

counter = 0
for cat_column in categorial_columns:
    value_counts = bank_X[cat_column].value_counts()
    
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    
    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index)
    
    axs[trace_x, trace_y].set_title(cat_column)
    
    for tick in axs[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(45)
    
    counter += 1

plt.show()