# %%
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
np.random.seed(42)

def minmax(row1, row2):
    row1 = row1.reshape(-1, 1)
    row2 = row2.reshape(-1, 1)
    # Ruzicka Distance, Weighted Jaccard distance
    q = np.concatenate([row1, row2], axis=1)
    # print(q, '\n\n')
    # print(np.amin(q,axis=1), '\n\n')
    # print(np.amax(q,axis=1), '\n\n')
    sim = np.sum(np.amin(q,axis=1))/np.sum(np.amax(q,axis=1))
    # print(sim)
    return 1 - sim


# Sample data
data = {f'doc{i}': {f'word{j}': np.random.randint(1, 10) for j in range(1, 21)} for i in range(1, 16)}

# Creating the DataFrame
df = pd.DataFrame.from_dict(data, orient='index')
df.to_csv('minmax.csv', index_label=False, index=True, header=True)
print(df, '\n\n')

mx = pd.DataFrame(pairwise_distances(df.values, metric=minmax), index=df.index, columns=df.index)
print(mx)


# %%
