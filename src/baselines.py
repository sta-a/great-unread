# %%
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import random
from collections import Counter

# %% [markdown]
# # Multiclass

# %%
def _get_f1_score(y, ypred):
    score = f1_score(y, ypred, average='macro')
    print(score)

# %%
n_eng = 556
zero_eng = 365
one_eng = 15
two_eng = 63
three_eng = 113

n_ger = 503
zero_ger = 330
one_ger = 10
two_ger = 86
three_ger = 77

y_eng = zero_eng*[0] + one_eng*[1] + two_eng*[2] + three_eng*[3]
y_ger = zero_ger*[0] + one_ger*[1] + two_ger*[2] + three_ger*[3]

# %% [markdown]
# #### Classify every data point as 0 /not reviewed  (majority class)

# %%
y_pred_eng = n_eng*[0]
_get_f1_score(y_eng, y_pred_eng)

y_pred_ger = n_ger*[0]
_get_f1_score(y_ger, y_pred_ger)

# %% [markdown]
# #### Assign each class with possibility 1/4

# %%
classes = [0,1,2,3]
y_pred_eng = random.choices(classes, k=n_eng)
Counter(y_pred_eng)
_get_f1_score(y_eng, y_pred_eng)

y_pred_ger = random.choices(classes, k=n_ger)
Counter(y_pred_ger)
_get_f1_score(y_ger, y_pred_ger)

# %% [markdown]
# #### Class probability = class frequency

# %%
classes = [0,1,2,3]
weights_eng = [zero_eng/n_eng, one_eng/n_eng, two_eng/n_eng, three_eng/n_eng]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_f1_score(y_eng, y_pred_eng)

weights_ger = [zero_ger/n_ger, one_ger/n_ger, two_ger/n_ger, three_ger/n_ger]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_f1_score(y_ger, y_pred_ger)

# %% [markdown]
# #### Class probability of two major classes

# %%
classes = [0,3]
weights_eng = [zero_eng/(zero_eng + three_eng), three_eng/(zero_eng + three_eng)]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_f1_score(y_eng, y_pred_eng)

classes = [0,2]
weights_ger = [zero_ger/(zero_ger + three_ger), two_ger/(zero_ger + two_ger)]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_f1_score(y_ger, y_pred_ger)

# %% [markdown]
# #### p(0) = 0.5, p(2)=p(3)=0.25

# %%
classes = [0,2,3]
weights_eng = [0.5, 0.25, 0.25]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_f1_score(y_eng, y_pred_eng)

weights_ger = [0.5, 0.25, 0.25]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_f1_score(y_ger, y_pred_ger)

# %% [markdown]
# # Accuracy

# %% [markdown]
# ## Twoclass 

# %%
def _get_accuracy(y, ypred):
    print(accuracy_score(y, ypred))

# %%
n_eng = 556
zero_eng = 365
one_eng = 191

n_ger = 503
zero_ger = 330
one_ger = 173

y_eng = zero_eng*[0] + one_eng*[1]
y_ger = zero_ger*[0] + one_ger*[1]

# %% [markdown]
# #### Classify every data point as 0 /not reviewed  (majority class)

# %%
y_pred_eng = n_eng*[0]
_get_accuracy(y_eng, y_pred_eng)

y_pred_ger = n_ger*[0]
_get_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Assign each class with possibility 1/2

# %%
classes = [0,1]
y_pred_eng = random.choices(classes, k=n_eng)
Counter(y_pred_eng)
_get_accuracy(y_eng, y_pred_eng)

y_pred_ger = random.choices(classes, k=n_ger)
Counter(y_pred_ger)
_get_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Class probability = class frequency

# %%
classes = [0,1]
weights_eng = [zero_eng/n_eng, one_eng/n_eng]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_accuracy(y_eng, y_pred_eng)

weights_ger = [zero_ger/n_ger, one_ger/n_ger]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# ## Library

# %%
n_eng = 603
zero_eng = 146
one_eng = 457

n_ger = 546
zero_ger = 240
one_ger = 306

y_eng = zero_eng*[0] + one_eng*[1]
y_ger = zero_ger*[0] + one_ger*[1]

# %% [markdown]
# #### Classify every data point as 1/featured (majority class)

# %%
y_pred_eng = n_eng*[1]
_get_accuracy(y_eng, y_pred_eng)

y_pred_ger = n_ger*[1]
_get_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Assign each class with possibility 1/2

# %%
classes = [0,1]
y_pred_eng = random.choices(classes, k=n_eng)
Counter(y_pred_eng)
_get_accuracy(y_eng, y_pred_eng)

y_pred_ger = random.choices(classes, k=n_ger)
Counter(y_pred_ger)
_get_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Class probability = class frequency

# %%
classes = [0,1]
weights_eng = [zero_eng/n_eng, one_eng/n_eng]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_accuracy(y_eng, y_pred_eng)

weights_ger = [zero_ger/n_ger, one_ger/n_ger]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# # Balanced Accuracy

# %% [markdown]
# ## Twoclass 

# %%
def _get_balanced_accuracy(y, ypred):
    print(balanced_accuracy_score(y, ypred))

zero_eng = 365
one_eng = 191
n_eng = 556

zero_ger = 332
one_ger = 171
n_ger = 503

y_eng = zero_eng*[0] + one_eng*[1]
y_ger = zero_ger*[0] + one_ger*[1]

# %% [markdown]
# #### Classify every data point as 0 /not reviewed  (majority class)

# %%
y_pred_eng = n_eng*[0]
_get_balanced_accuracy(y_eng, y_pred_eng)

y_pred_ger = n_ger*[0]
_get_balanced_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Assign each class with possibility 1/2

# %%
classes = [0,1]
y_pred_eng = random.choices(classes, k=n_eng)
print(Counter(y_pred_eng))
_get_balanced_accuracy(y_eng, y_pred_eng)

y_pred_ger = random.choices(classes, k=n_ger)
print(Counter(y_pred_ger))
_get_balanced_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Class probability = class frequency

# %%
classes = [0,1]
weights_eng = [zero_eng/n_eng, one_eng/n_eng]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_balanced_accuracy(y_eng, y_pred_eng)

weights_ger = [zero_ger/n_ger, one_ger/n_ger]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_balanced_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# ## Library

# %%
zero_eng = 146
one_eng = 457
n_eng = 603

zero_ger = 246
one_ger = 300
n_ger = 546

y_eng = zero_eng*[0] + one_eng*[1]
y_ger = zero_ger*[0] + one_ger*[1]

# %% [markdown]
# #### Classify every data point as 1/featured (majority class)

# %%
y_pred_eng = n_eng*[1]
_get_balanced_accuracy(y_eng, y_pred_eng)

y_pred_ger = n_ger*[1]
_get_balanced_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Assign each class with possibility 1/2

# %%
classes = [0,1]
y_pred_eng = random.choices(classes, k=n_eng)
Counter(y_pred_eng)
_get_balanced_accuracy(y_eng, y_pred_eng)

y_pred_ger = random.choices(classes, k=n_ger)
Counter(y_pred_ger)
_get_balanced_accuracy(y_ger, y_pred_ger)

# %% [markdown]
# #### Class probability = class frequency

# %%
classes = [0,1]
weights_eng = [zero_eng/n_eng, one_eng/n_eng]
y_pred_eng = random.choices(classes, weights_eng, k=n_eng)
_get_balanced_accuracy(y_eng, y_pred_eng)

weights_ger = [zero_ger/n_ger, one_ger/n_ger]
y_pred_ger = random.choices(classes, weights_ger, k=n_ger)
_get_balanced_accuracy(y_ger, y_pred_ger)

# %%


# %%



