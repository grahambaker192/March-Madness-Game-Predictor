import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import csv
import pandas as pd
import numpy as np
loaded_model = tf.keras.models.load_model("my_model.keras")
current_stats = pd.read_csv('currentStats.csv')
#pulled_stats = ['EFG_O','TOR', 'ORB', 'DRB', 'FTR', '3P_O', 'SEED']
pulled_stats = ['FG%', 'TOV', 'ORB', 'DRB', 'FTA']
current_stats = current_stats[pulled_stats]
current_stats=np.array(current_stats)
Ucon = current_stats[306]
SanDiego = current_stats[51]
Illinois = current_stats[195]
IowaSt =current_stats[298]
NorthCarolina = current_stats[130]
Alabama = current_stats[187]
Clemson = current_stats[117]
Arizona = current_stats[44]
Houston = current_stats[108]
Duke = current_stats[342]
NCstate = current_stats[59]
Marquett = current_stats[299]
Purdue = current_stats[106]
Gonzaga = current_stats[340]
Creighton = current_stats[5]
Tennnessee = current_stats[290]
team1 = np.array(Ucon)
team2 = np.array(Houston)
temp = Ucon
test = np.divide(team1, team2)


test = test.reshape(1, -1)
#Effective Field Goal Percentage Shot, Effective Field Goal Percentage Allowed, Turnover Rate, Defensve Rebounding Rate,  Free Throw Rate, Two-Point Shooting Percentage, Three-Point Shooting Percentage, SEED
# Make predictions using the trained model

print(test.shape)
predictions = loaded_model.predict(test)

# The output 'predictions' will contain the model's predictions for the input data point
print("Predictions:", predictions)