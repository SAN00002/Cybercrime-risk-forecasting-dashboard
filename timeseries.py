import pandas as pd

# Load data
df = pd.read_csv("enhanced_cybercrime.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Aggregate total loss per year
global_yearly_loss = df.groupby('year')['financial_loss_(in_million_$)'].sum().reset_index()

# Prophet needs: ds = date, y = target
global_yearly_loss.columns = ['ds', 'y']

# Convert year to datetime
global_yearly_loss['ds'] = pd.to_datetime(global_yearly_loss['ds'], format='%Y')
from prophet import Prophet

model = Prophet()
model.fit(global_yearly_loss)
# Forecast 10 future years
future = model.make_future_dataframe(periods=10, freq='Y')
forecast = model.predict(future)
import matplotlib.pyplot as plt

model.plot(forecast)
plt.title("Forecasted Global Financial Loss from Cybercrime")
plt.xlabel("Year")
plt.ylabel("Predicted Loss (in $ millions)")
plt.show()
# Group per country-year
by_country = df.groupby(['country', 'year'])['financial_loss_(in_million_$)'].sum().reset_index()

# Forecast for a single country (e.g., India)
country_name = "India"
subset = by_country[by_country['country'] == country_name].copy()
subset.columns = ['country', 'ds', 'y']
subset['ds'] = pd.to_datetime(subset['ds'], format='%Y')

# Fit Prophet
model = Prophet()
model.fit(subset[['ds', 'y']])

future = model.make_future_dataframe(periods=10, freq='Y')
forecast = model.predict(future)

model.plot(forecast)
plt.title(f"{country_name} - Forecasted Cybercrime Loss")
plt.show()
