import pandas as pd
import numpy as np
df= pd.DataFrame(np.random.randn (5,3))
index = ['a','c','e','f','h']
data = [{'a':1,'b':2}]
columns = ['one','two','three']
df = df.reindex(['a','b','c','d','e','f','g','h'])
print (df)
