import numpy as np
from sklearn.metrics import f1_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(f1_score(y_true, y_pred, average='macro'),
      f1_score(y_true, y_pred, average='micro'),
      f1_score(y_true, y_pred, average='weighted'),
      f1_score(y_true, y_pred, average=None))
# %%
# binary classification
y_true_empty = [0, 0, 0, 0, 0, 0]
y_pred_empty = [0, 0, 0, 0, 0, 0]
print(f1_score(y_true_empty, y_pred_empty),
      f1_score(y_true_empty, y_pred_empty, zero_division=1.0),
      f1_score(y_true_empty, y_pred_empty, zero_division=np.nan))
# %%
# multilabel classification
y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
f1_score(y_true, y_pred, average=None)

