import pandas as pd

# initialize a dictionary to store the data
trial_data = []

# read the input text from a file
with open('slurm-385012.out', 'r') as file:
    current_trial_data = {}
    # iterate over the lines
    for line in file:
        # check if the line contains hyperparameters data
        if "Trial #" in line:
            if current_trial_data:  # if current_trial_data is not empty, append it to trial_data
                trial_data.append(current_trial_data)
            # extract trial number
            current_trial_data = {"Trial": int(line.split('#')[1])}
        elif "|" in line and "Best Value" not in line and "tuner" not in line:
            value, _, hyperparameter = line.split("|")
            current_trial_data[hyperparameter.strip()] = float(value.strip())
        #elif 'val_loss:' in line and 'Trial' not in line:
        #    current_trial_data['val_loss'] = float(line.split("val_loss:")[1].strip())
        elif 'loss:' in line and 'val_loss' in line:
            current_trial_data['loss'] = float(line.split("loss:")[1].split(" - ")[0].strip())
            current_trial_data['val_loss'] = float(line.split("val_loss:")[1].split(" - ")[0].strip())
    # don't forget to append the last trial's data
    trial_data.append(current_trial_data)

# create the hyperparameters dataframe
df_hyperparameters = pd.DataFrame(trial_data)

# print the dataframe
print(df_hyperparameters)
df_hyperparameters.to_csv('tunning_score.csv')