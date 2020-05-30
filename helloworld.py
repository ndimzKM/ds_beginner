from sklearn.datasets import load_iris

iris = load_iris()

for i in range(len(iris.target)):
    print('Example %d: Label: %d features: %s ' % (i, iris.target[i], iris.data[i]))