'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages

import pandas as pd


# Your code here

# loading the raw CSVs
pred_universe = pd.read_csv('data/pred_universe_raw.csv')
arrest_events = pd.read_csv('data/arrest_events_raw.csv')

# converting arrest date columns to datetime for both datasets
pred_universe['arrest_date_univ'] = pd.to_datetime(pred_universe['arrest_date_univ'])
arrest_events['arrest_date_event'] = pd.to_datetime(arrest_events['arrest_date_event'])

# merging on person_id with a full outer join
df_arrests = pd.merge(pred_universe, arrest_events, how='outer', on='person_id', suffixes=('_univ', '_event'))

# filtering felony arrests from arrest_events so that i can use it later
felony_events = arrest_events[arrest_events['charge_degree'] == 'F']

# checking if a person was rearrested for felony within 365 days after their current arrest date
def was_rearrested(person_id, arrest_date_event):
    future = felony_events[
        (felony_events['person_id'] == person_id) &
        (felony_events['arrest_date_event'] > arrest_date_event) &
        (felony_events['arrest_date_event'] <= arrest_date_event + pd.Timedelta(days=365))
    ]
    return 1 if not future.empty else 0

df_arrests['y'] = df_arrests.apply(
    lambda row: was_rearrested(row['person_id'], row['arrest_date_event']) if pd.notnull(row['arrest_date_event']) else 0,
    axis=1
)

rearrest_rate = df_arrests['y'].mean()
print(f"What share of arrestees in df_arrests were rearrested for a felony in the next year? {rearrest_rate:.2%}")

# making current_charge_felony: 1 if current charge is felony, 0 otherwise
df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'F' else 0)
felony_charge_share = df_arrests['current_charge_felony'].mean()
print(f"What share of current charges are felonies? {felony_charge_share:.2%}")

# counting number of felony arrests in the 365 days *before* current arrest date
def count_prior_felonies(person_id, arrest_date_event):
    if pd.isnull(arrest_date_event):
        return 0
    prior = felony_events[
        (felony_events['person_id'] == person_id) &
        (felony_events['arrest_date_event'] >= arrest_date_event - pd.Timedelta(days=365)) &
        (felony_events['arrest_date_event'] < arrest_date_event)
    ]
    return len(prior)

df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(
    lambda row: count_prior_felonies(row['person_id'], row['arrest_date_event']),
    axis=1
)

avg_prior_felonies = df_arrests['num_fel_arrests_last_year'].mean()
print(f"What is the average number of felony arrests in the last year? {avg_prior_felonies:.2f}")

print(f"Mean of num_fel_arrests_last_year: {df_arrests['num_fel_arrests_last_year'].mean():.2f}")

print(df_arrests.head())

# saving data 
df_arrests.to_csv('data/df_arrests.csv', index=False)

