import pandas as pd

input_data_df = pd.read_csv("data/time_prediction_input.csv", sep='\t')

input_data_df['start_year'] = input_data_df['start_date'].str[-4:]
input_data_df['completion_year'] = input_data_df['completion_date'].str[-4:]

# shape of train_df: (77815, 10)
# shape of test_df: (36466, 10)

train_df = input_data_df[input_data_df['completion_year'] < '2019']
test_df = input_data_df[input_data_df['start_year'] >= '2019']

train_df.drop(['start_year', 'completion_year'], axis=1, inplace=False).to_csv("data/time_prediction_train.csv", sep='\t', index=False)
test_df.drop(['start_year', 'completion_year'], axis=1, inplace=False).to_csv("data/time_prediction_test.csv", sep='\t', index=False)

print(train_df.shape)
print(test_df.shape)
