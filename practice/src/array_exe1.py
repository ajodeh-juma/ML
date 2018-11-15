#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pandas as pd
import tempfile, os.path
from mlxtend.preprocessing import standardize
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

X = np.random.randn(4,2) # random normals in 4x2 array
print (X)
print ('\n')


# For each column find the row index of the minimum value

#print (np.argwhere(X==np.min(X)), np.min(X))
#print (np.where(X==np.min(X)), np.min(X))

for idx, col in enumerate(X.T):
	print ("minimum value is: %f at index %d" %  (min(col), np.argmin(col)))

#function that performs column-based standardization on a NumPy array
# the result of standardization (Z-score normalization) is that the features\
# will be rescaled so that they'll have the properties of a standard normal distribution
# with 

plt.title(r'$\alpha > \beta$')

plt.title(r"\TeX\ is Number "
          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
          fontsize=16, color='gray')


X_std, params = standardize(X, columns=[0,1], return_params=True)
print (X_std)
print (params)

X_df=pd.DataFrame(X, columns=['a','b'])
xdf_std=standardize(X_df, columns=['a','b'])
print (xdf_std)


# data manipulation - pandas

print("data manipulation using pandas\n")

columns=['name', 'age', 'gender', 'job']
user1=pd.DataFrame([['alice', 19, 'F', 'student'],['john', 26, 'M', 'student']], columns=columns)
user2=pd.DataFrame([['eric', 22, 'M', 'student'], ['paul', 58, 'F', 'manager']], columns=columns)
user3=pd.DataFrame(dict(name=['peter','julie'], age=[33, 44], gender=['M', 'F'], job=['engineer', 'scientist']))

#concatenate
user1.append(user2)
print(user1)
users=pd.concat([user1,user2,user3])
print (users)

#join
user4=pd.DataFrame(dict(name=['alice', 'john', 'eric', 'julie'], height=[165, 180, 175, 171]))

print (user4)
# use intersection of keys from both frames

merger_inter=pd.merge(users, user4, on="name")
print (merger_inter)

#use union of keys from both frames
users=pd.merge(users, user4, on="name", how="outer")
print(users)

print("beginning data manipulation using pandas exercises\n")

#url="https://raw.github.com/neurospin/pystatsml/master/data/iris.csv"
#data=pd.read_csv(url)
#df=pd.DataFrame(data)

#print (df.columns) #column names
#for grp, sp in df.groupby("species"):
#	print (grp, sp.mean())

df=users.copy()
print ('\n')

#for i in range(df.shape[0]):
#	df.ix[i, 'age'] *= 10
#print (df)
#print('\n')

age_job=users[users.age<20]
print(age_job)
print('\n')

print(users[(users.age>20) & (users.gender=='M')])
print(users.job.isin(['student','engineer']))

#sorting
print("\nsorting\n")
print(df.sort_values(by='age', ascending=False))
print('\n')
print(df.sort_values(by=['job','age']))

print("\nquality control: duplicate data\n")
df=users.append(df.iloc[0], ignore_index=True)
print(df[df.duplicated()])
df=df.drop_duplicates()
print("\nquality control: missing data\n")

print(df.describe(include='all')) # exclude all missing values

print (df.isnull()) # DataFrame of booleans
print(df.isnull().sum()) # calculate the sum of each column
print('\n')

#strategy 1: drop missing values
print(df.dropna()) # drop a row if ANY values are missing
print(df.dropna(how='all')) # drop a row only if ALL values are missing 

#strategy 2: fill in missing values
print(df.height.mean())
df=users.copy()
df.ix[df.height.isnull(),'height']=df['height'].mean()
print('\n--------\n')
print(df)

print('\n------rename values-------\n')

df=users.copy()
print(df.columns)
df.columns=['age','genre','travail','nom','taille']
df.travail=df.travail.map({'student':'etudent', 'manager':'manager', 'engineer':'inginieur',
	'scientist':'scientific'})
assert df.travail.isnull().sum()==0
print(df)

print('\n------dealing with outliers ------\n')

size=pd.Series(np.random.normal(loc=175, size=20, scale=10))
print(size[:3])

print('\n------File I/O-------\n')
print("\n----csv-----\n")

tmpdir=tempfile.gettempdir()
csv_filename=os.path.join(tmpdir,'users.csv')
users.to_csv(csv_filename, index=False)
other=pd.read_csv(csv_filename)


print("\n----Excel-----\n")

xls_filename=os.path.join(tmpdir,'users.xlsx')
users.to_excel(xls_filename, sheet_name='users', index=False)
xls_df=pd.read_excel(xls_filename, sheet_name='users')

print ('\n---------Multiple sheets-------\n')
with pd.ExcelWriter(xls_filename) as writer:
	users.to_excel(writer, sheet_name='users', index=False)
	df.to_excel(writer, sheet_name='salary', index=False)


df=users.copy()
df.ix[[0,2],'age']=None
df.ix[[1,2],'gender']=None
print(df)

def fillmissing_with_mean(df):
	"""fill all missing values of numerical column with the  mean of the current columns"""
	imputed=df.fillna(df.mean()).dropna(axis=1, how='all')
	
	with pd.ExcelWriter(xls_filename) as writer:
		df.to_excel(writer, sheet_name='original', index=False)
		imputed.to_excel(writer, sheet_name='imputed', index=False)
	return imputed

print("\n-----Basic plots------\n")
x=np.linspace(0,10,50)
sinus=np.sin(x)
cosinus = np.cos(x)
plt.plot(x, sinus, label='sinus', color='blue', linestyle='--', linewidth=2)
plt.plot(x, cosinus, label='cosinus', color='red', linestyle='-', linewidth=2)
plt.legend()
plt.savefig("sinus-cosine.svg")
plt.close()




if __name__ == '__main__':
	df=users.copy()
	df.ix[[0,2],'age']=None
	df.ix[[1,2],'gender']=None
	fill=fillmissing_with_mean(df)

