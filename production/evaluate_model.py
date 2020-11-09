import json
import math
import pickle

import pandas as pd
from config import Config
from sklearn.metrics import mean_squared_error, r2_score

# Reading in the test datasets
x_test = pd.read_csv(str(Config.features_path / 'test_features.csv'))
y_test = pd.read_csv(str(Config.features_path / 'test_target.csv'))

# Loading our model we saved earlier
model = pickle.load(open(str(Config.models_path / 'model.pickle'), 'rb'))

# Calculating the coefficent of determination (r²)
r_squared = model.score(x_test, y_test)

# Performing predictions
y_pred = model.predict(x_test)

# Calculating root mean squared error
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
coefficent_r2 = r2_score(y_test, y_pred)

# Saving metrics in a json file
with open(str(Config.metrics_file_path), 'w') as outfile:
    json.dump(dict(r_squared=coefficent_r2, rmse=rmse), outfile)
