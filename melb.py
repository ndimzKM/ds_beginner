import pandas as pd

melbourne_file_path = './melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)

melbourne_data.describe()
#print(melbourne_data.columns)
y = melbourne_data.Price

melbourne_features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']

X = melbourne_data[melbourne_features]

#print(X.head())
#print(y)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X,val_X, train_y, val_y):
	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
	model.fit(train_X, train_y)
	preds_val = model.predict(val_X)
	mae = mean_absolute_error(val_y, preds_val)
	return mae

for max_leaf_nodes in [5, 50, 500, 5000]:
	my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
	print("My leaf nodes: %d \t\t Mean Absolute Error: %d" % (max_leaf_nodes, my_mae))
"""
melbourne_model = DecisionTreeRegressor()

melbourne_model.fit(train_X, train_y)

from sklearn.metrics import mean_absolute_error

val_predicted = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predicted))
"""
#print("Making predtictions for the following 5 houses:")
#print(X.head())
#print("The predictions are:")
#print(melbourne_model.predict(X.head()))
