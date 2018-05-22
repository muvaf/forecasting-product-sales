from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
np.random.seed(7)
df = pd.read_csv("dataset/train.csv", sep=',', parse_dates=[2])

df = df[df['Store'] == 1.0][df['Date']>datetime.date(2013,1,6)].sort_values(by='Date')
df = df[df['DayOfWeek'] != 7]

df['SalesMinus1'] = df['Sales'].shift(1)
df['SalesMinus2'] = df['Sales'].shift(2)
df['CustomersMinus1'] = df['Customers'].shift(1)
df['CustomersMinus2'] = df['Customers'].shift(2)
df = df.dropna()
df = df.drop(['Customers', 'StateHoliday', 'Store', 'Date'], axis = 1)
df = pd.get_dummies(df, columns=['DayOfWeek'])

'''
Data Preperation
'''
import numpy as np
from sklearn.model_selection import train_test_split
labels = np.array(df['Sales'])
df_nosales = df.drop('Sales', axis = 1)
feature_list = list(df_nosales.columns)
np_data = np.array(df_nosales)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(np_data, labels, test_size = 0.25, random_state = 42)



def sales_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)
estimator.fit(train_features, train_labels)

'''
Prediction
'''
from matplotlib import pyplot as plt

predictions = estimator.predict(test_features)
# Fix for divide by 0 problem.
test_labels[test_labels == 0] = np.mean(test_labels)
predictions[predictions == 0] = np.mean(predictions)

errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
plt.plot(range(0, len(predictions)), predictions, label='predictions')
plt.plot(range(0, len(test_labels)), test_labels, label='actual values')
plt.legend()

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
