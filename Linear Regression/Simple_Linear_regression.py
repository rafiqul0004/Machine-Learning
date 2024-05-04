import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes=datasets.load_diabetes()
# diabetes_x=diabetes.data[:,np.newaxis,2]
diabetes_x=diabetes.data
#features
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-30:]

#levels
diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()

# fit the model with the training data
model.fit(diabetes_x_train,diabetes_y_train)

diabetes_y_predict=model.predict(diabetes_x_test)
 #Mean Squared
print("Mean squared error: ", mean_squared_error(diabetes_y_test,diabetes_y_predict))

print("weight : ", model.coef_)
print("Intercept : ", model.intercept_)

#plotting
# plt.scatter(diabetes_x_test,diabetes_y_test)
# plt.plot(diabetes_x_test,diabetes_y_predict)
# plt.show()