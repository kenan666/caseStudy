import pandas as pd 
import numpy as np

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url,sep = '\t')

chipo.head(10)

#  数据集中的观测数量是多少？
chipo.shape[0]

chipo.columns
#  Print the name of all the columns
chipo.index

#  Which was the most-ordered item?
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'],ascending = False)
c.head(1)

#  For the most-ordered item, how many items were ordered?
c = chipo.groupby('choice_description').sum()
c = c.sort_values(['quantity'],ascending = False)
c.head(1)

#  How many items were orderd in total?
total_items_orders = chipo.quantity.sum()
total_items_orders

#  Turn the item price into a float
chipo.item_price.dtype

'''
,lambda作为一个表达式，定义了一个匿名函数
例：
g = lambda x:x+1
g(1) -> 2    g(2)->3  

类似于：  x  作为函数入口   x+1作为函数体
def g(x):
    return x+1

lambda  简化了函数定义的书写形式，使代码简洁，但是函数定义方式更加直观，易理解

'''

dollraizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollraizer)

chipo.item_price.dtype

# How much was the revenue for the period in the dataset?
revenue = (chipo['quantity'] * chipo['item_price']).sum()
print('Revenue was: $' + str (np.round(revenue,2)))

#  How many orders were made in the period?

orders = chipo.order_id.value_counts().count()
orders

#  What is the average revenue amount per order?
chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by = ['order_id']).sum()
order_grouped.mean()['revenue']

# How many different items are sold?
chipo.item_name.value_counts().count()