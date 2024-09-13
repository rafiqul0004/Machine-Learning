#Loading Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading Dataset
iris=datasets.load_iris()

#Printing Description
# print(iris.DESCR)
features = iris.data
lebels=iris.target
# print(features[0],lebels[0])

#Loading Classifier
clf=KNeighborsClassifier()
clf.fit(features,lebels)

preds=clf.predict([[5.1,2,1,100]])
print(preds)