import pandas as pd
import matplotlib.pyplot as plt
import sys
#create dataframe
df=pd.read_csv('datafile1.csv', header=None, names=['col1','col2','col3','col4','col5','col6'])

#plot scatter with first column as x values and second column as y values
plt.scatter(df['col3'],df['col4'],color='red',label="Authorized Capital - Private")
plt.scatter(df['col2'],df['col5'],color='blue',label="No. of Companies - Total")
#specifying labels
plt.xlabel("x-label")
plt.ylabel("y-label")

#enable legend
plt.legend()
plt.show()
