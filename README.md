



### Name: A.Sasidharan
### Register no:212221240049
### Date:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
~~~

import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/globaltemper.csv', parse_dates=['dt'], index_col='dt')

# Assuming 'AverageTemperature' is the column with temperature data
temperature_data = data['AverageTemperature']

# Plot the temperature data
plt.figure(figsize=(10, 6))
plt.plot(temperature_data.index, temperature_data, label='Global Temperature Change')  # Plot index vs. values
plt.title('Global Temperature Change Data')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()

# Perform ADF test
adf_result = adfuller(temperature_data)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

# Split data into train and test sets
train_size = int(len(temperature_data) * 0.8)  # Use temperature_data for length
train, test = temperature_data.iloc[:train_size], temperature_data.iloc[train_size:]

# Adjust lag order to be less than the number of available data points
# Reduced lag order to 5 (you might need to experiment with this value)
lag_order = 5  

# Fit AutoReg model with the adjusted lag order
model = AutoReg(train, lags=lag_order)  
model_fitted = model.fit()


# Plot ACF and PACF with adjusted lags
plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(temperature_data, lags=len(temperature_data) // 2 -1, ax=plt.gca())  # Use temperature_data for ACF, lags adjusted to be less than or equal to 50% of data length
plt.title('Autocorrelation Function (ACF)')

plt.subplot(212)
plot_pacf(temperature_data, lags=len(temperature_data) // 2 - 1, ax=plt.gca())  # Use temperature_data for PACF, lags adjusted to be less than or equal to 50% of data length
plt.title('Partial Autocorrelation Function (PACF)')


plt.tight_layout()
plt.show()

# Make predictions
predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot predictions vs. actual test data
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Test Data', color='blue')
plt.plot(test.index, predictions, label='Predicted Data', color='red')
plt.title('Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Average Temperature') 
plt.legend()
plt.show()

# Calculate and print MSE
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')



# Make final prediction
final_prediction = model_fitted.predict(start=len(temperature_data), end=len(temperature_data))
print(f'Final Prediction for Next Time Step: {final_prediction[len(temperature_data)]}')

~~~
### OUTPUT:

GIVEN DATA


![image](https://github.com/user-attachments/assets/c8617b19-47f3-417e-bfe3-ad07ca71394c)


PACF - ACF

![image](https://github.com/user-attachments/assets/b1e9c251-e6b9-4202-bc33-a127359d7ac5)


PREDICTION
![image](https://github.com/user-attachments/assets/42156a10-efe6-4f96-986a-6d39de34b97f)




FINIAL PREDICTION

![image](https://github.com/user-attachments/assets/dd91bf0e-1902-4c3d-b959-139938ab9278)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
