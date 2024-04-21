import numpy as np
import pandas as pd
import sklearn as sk

train = pd.read_csv('2024_DS2_HW1_data_train.csv')
test = pd.read_csv('2024_DS2_HW1_data_test.csv')

'''
print(train.head())
print(train.shape)
print(test.shape)
'''

###
# data preparation
###

# NaN imputation
train = train.fillna(train.median())

# getting rid of too specific predictors
# i.e. only keep those where all levels contain >= c
# or those with more than l levels, as they are probably numeric
drops = []
c = 0.01
l = 10
for col in train.columns:
    col_portions = train[col].value_counts(normalize=True)
    
    print("Column:", col)
    print("Proportions of Observations for Each Level:")
    print(col_portions)
    
    if np.min(col_portions) < c and train[col].nunique() < l: drops.append(col)

'''    
print(train.shape)
print(drops)
'''
# we have extracted some predictors which contain lots of values
# with relatively small representations
# it will probably be sufficient to merge the small categories into one larger category
train['adults'] = train['no_of_adults'].apply(lambda x: x if x in [1, 2] else 'other')
train['children'] = train['no_of_children'].apply(lambda x: 0 if x == 0 else '1+')
train['weekend_nights'] = train['no_of_weekend_nights'].apply(lambda x: x if x in [0, 1, 2, 3, 4, 5, 6] else '7+')
train['room_type'] = train['room_type_reserved'].apply(lambda x: x if x in ['Room_Type 1', 'Room_Type 4', 'Room_Type 6'] else 'Other')
train['segment'] = train['market_segment_type'].apply(lambda x: x if x in ['Online', 'Offline', 'Corporate'] else 'Other')
train['meal_plan'] = train['type_of_meal_plan'].apply(lambda x: x if x in ['Meal Plan 1', 'Not Selected'] else 'Other')
train['cancel'] = train['no_of_previous_cancellations'].apply(lambda x: 0 if x==0 else '1+')
train['special'] = train['no_of_special_requests'].apply(lambda x: x if x in [0, 1, 2] else '3+')

train = train.drop(columns=drops)

# now do the same for test set
test['adults'] = test['no_of_adults'].apply(lambda x: x if x in [1, 2] else 'other')
test['children'] = test['no_of_children'].apply(lambda x: 0 if x == 0 else '1+')
test['weekend_nights'] = test['no_of_weekend_nights'].apply(lambda x: x if x in [0, 1, 2, 3, 4, 5, 6] else '7+')
test['room_type'] = test['room_type_reserved'].apply(lambda x: x if x in ['Room_Type 1', 'Room_Type 4', 'Room_Type 6'] else 'Other')
test['segment'] = test['market_segment_type'].apply(lambda x: x if x in ['Online', 'Offline', 'Corporate'] else 'Other')
test['meal_plan'] = test['type_of_meal_plan'].apply(lambda x: x if x in ['Meal Plan 1', 'Not Selected'] else 'Other')
test['cancel'] = test['no_of_previous_cancellations'].apply(lambda x: 0 if x==0 else '1+')
test['special'] = test['no_of_special_requests'].apply(lambda x: x if x in [0, 1, 2] else '3+')

test = test.drop(columns=drops)