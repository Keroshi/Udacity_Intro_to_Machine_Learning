from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=100, criterion='log_loss')
clf.fit(X, Y)
pred = clf.predict([[0, 0], [1, 1]])
accuracy = accuracy_score([0, 1], pred)
print(accuracy)
print(pred)
