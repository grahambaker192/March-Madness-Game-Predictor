import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import csv
import pandas as pd
import numpy as np

pulled_stats = ['EFG_O','TOR', 'ORB', 'DRB', 'FTR'] #Stats we want to look at
df = pd.read_csv('results.csv') # Reads March Madness results CSV
results_array = df.values #Truns results.csv into an array
stats = pd.read_csv('cbb.csv')# Reads March Madness D1 team stats
team_names = stats['TEAM'] # Holds all the Team names
team_year = stats['YEAR'] # Holds all the Teams year
pulled_stats_arr = stats[pulled_stats] # Pulls data we want to look at
stats_arr = pulled_stats_arr.values # Turns it into an accessable array
dates = []
X = [] # The array that will hold the lists of every game and every stat for each team (Team/Opponent)
Y = [] # This array will hol weater the team won or lost

#Pulls the year the game was  played
for date_string in results_array[:, 0]:
    year = date_string.split('/')[-1]
    if(int(year)>30):
        dates.append("19"+year)
    else:
        dates.append("20"+year)
#Pulls team data from relevent year
for index, value in enumerate(results_array):
    loser_idx = None
    winner_idx = None
    for num, name in enumerate(team_names):
        temp_year = team_year[num]
        if(int(temp_year)== int(dates[index])):
            if(name == value[4]):
                winner_idx = num
            if(name == value[7]):
                loser_idx = num

    temp_winner=[]
    temp_loser = []
    # adds winner/loser and loser/winner relationship to our dataset with corresponding Y value 0 (Correct) 1(Incorrect)
    if((loser_idx!=None)&(winner_idx!=None)):
        for idx, win_stat_val in enumerate(stats_arr[winner_idx]):
            temp_winner.append(win_stat_val/stats_arr[loser_idx][idx])
            temp_loser.append(stats_arr[loser_idx][idx]/win_stat_val)
        X.append(temp_winner)  # add stats relationship
        Y.append(0)  # Winner on top
        X.append(temp_loser)  # add stats relationship
        Y.append(1)  # Loser on top

#Splits our data set into train and tes
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
input_shape = (len(pulled_stats),)  # Assuming pulled_stats contains your input features
#Defien and create model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Turn sets into Numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Verify data types and shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#Train the model 
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model's performance on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy {accuracy}")
print(f"Loss {loss}")

#Save the model for further use
model.save('my_model.keras')
