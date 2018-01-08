import pandas as pd
import matplotlib.pyplot as plt
import sys
#create dataframe from csv
df=pd.read_csv('datafile1.csv', header=None, names=['col1','col2','col3','col4','col5','col6'])
plotMap=[]

#create a list of lists where each list will have a corresponding box plot
plotMap.append(df['col2'].dropna().tolist())
plotMap.append(df['col4'].dropna().tolist())

#plotting
plt.boxplot(plotMap)
#plt.boxplot("box plot for economic activity")
plt.yscale('log')


#ax.set_yscale('log')

#specifying labels
plt.xticks([1,2],[" Real Estate and Renting"," Community"])
plt.xlabel("x-label")
plt.ylabel("y-label")


plt.legend()
plt.show()
