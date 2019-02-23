import pandas as pd

users = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', 
                      sep='|', index_col='user_id')

'''
head()    头部 n 个元素
tail()    尾部n 个元素
'''
users.head(25)   
users.tail(10) 

#  What is the number of observations in the dataset?
users.shape[0]

#  What is the number of columns in the dataset?
users.shape[1]

#  print the name of all columns
users.columns

#  how is the dataset indexed
users.index

#  what is the data type of each columns
users.dtypes

#  Print only the occupation column
users.occupation
# users['occupation']

# How many different occupations there are in this dataset?
users.occupation.nunique()

#  What is the most frequent occupation?
users.occupation.value_counts().head()

#  Summarize the DataFrame.

users.describe()

#  Summarize all the columns
users.describe(include = 'all')

# Summarize only the occupation column
users.occupation.describe()

#  What is the mean age of users?
#  用户的平均年龄是多少？
round(users.age.mean())  #  mean() 函数  对所有元素求均值

#  What is the age with least occurrence?

#  最少出现的年龄是多少？
users.age.value_counts().tail()