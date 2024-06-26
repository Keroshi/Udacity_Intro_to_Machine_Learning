import pandas as pd
import numpy as np
import seaborn as sns

# %%
df = sns.load_dataset('penguins')
df.head()
# %%
df.shape
# %%
df.info()
# %%
df.isnull().sum()
# %%
df.dropna(inplace=True)
# %%
df.isnull().sum()
# %%
df.sex.unique()
# %%
pd.get_dummies(df['sex']).head()
# %%
sex = pd.get_dummies(df['sex'], drop_first=True)
sex.head()

# %%
df.island.unique()
# %%
pd.get_dummies(df['island']).head()
# %%
island = pd.get_dummies(df['island'], drop_first=True)
island.head(5)
# %%
new_data = pd.concat([df, island, sex], axis=1)
# %%
new_data.head()
# %%
new_data.drop(['sex', 'island'], axis=1, inplace=True)
# %%
new_data.head()
# %%
Y = new_data.species
Y.head()
# %%
Y.unique()
# %%
Y = Y.map(
    {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})  # Using map function to convert categorical values to numerical values
Y.head()
# %%
new_data.drop('species', axis=1, inplace=True)
new_data.head()
# %%
X = new_data
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# %%
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# %%
y_pred = classifier.predict(X_test)
y_pred
# %%
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
# %%
accuracy_score(y_test, y_pred)
# %%
print(classification_report(y_test, y_pred))
# %%
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=7, criterion='gini', random_state=0)
classifier.fit(X_train, y_train)
# %%
y_pred = classifier.predict(X_test)
# %%
accuracy_score(y_test, y_pred)
