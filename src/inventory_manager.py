import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class InventoryManager:

    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def preprocess_data(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.index = self.data['date']
        self.data.drop(['date'], axis=1, inplace=True)

    def predict_demand(self, product_id):
        product_data = self.data[self.data['product_id'] == product_id]
        X = product_data[['features']].values  # Replace 'features' with actual feature column names
        y = product_data['demand'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        return model.predict(X_test)

# To use the inventory manager
# manager = InventoryManager('data/inventory_data.csv')
# manager.preprocess_data()
# demand_prediction = manager.predict_demand(product_id=101)
# print(demand_prediction)
