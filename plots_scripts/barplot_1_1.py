import pandas as pd
import matplotlib.pyplot as plt
import sys
#reading data frame from a csv file
df=pd.read_csv('datafile1.csv', header=None, names=['col1','col2','col3','col4','col5','col6'])
#plot bar plot with xticks which is position of bars as first argument and height of bars as second argument
plt.bar([1,2,3,4,5,6,7,8,9],df['col3'],color='darkgreen',label="bar-plot for economic activity")

#specify labels on xticks
plt.xticks([1,2,3,4,5,6,7,8,9],["Business Services"," Real Estate and Renting"," Trading"," Community"," Transport"," Finance"," Insurance","  Unclassified Total"," Total"])
plt.xlabel("x-label")
plt.ylabel("y-label")

#enabling legend
plt.legend()
plt.show()
