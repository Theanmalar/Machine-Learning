import pandas as pd
import numpy as np
df= pd.DataFrame(np.random.randn (5,3),
index = ['a','c','e','f','h'],
columns = ['one','two','three'])
df = df.reindex(['a','b','c','d','e','f','g','h'])
print (df)
print (df['one'].isnull())
print (df['one'].notnull())
print (df['one'].bfill())
print (df['one'].pad())
print (df['one'].fillna(0))
df = pd.DataFrame({'one':[1000,20,30,40,50,60],'two':[30,80,90,2000,70,100]})
print (df.replace({1000:10,2000:60}))
print (df)
