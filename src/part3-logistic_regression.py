'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression 


# Your code here

df_arrests = pd.read_csv('data/df_arrests.csv')

#defining features and target
features = ['num_fel_arrests_last_year', 'current_charge_felony']
target = 'y'

# splitting data into train and test sets
df_arrests_train, df_arrests_test = train_test_split(
    df_arrests,
    test_size=0.3,
    shuffle=True,
    stratify=df_arrests[target],
    random_state=42
)

# preparing X and y for training and testing
X_train = df_arrests_train[features]
y_train = df_arrests_train[target]

X_test = df_arrests_test[features]
y_test = df_arrests_test[target]

# defining parameter grid for C (regularization strength)
param_grid = {'C': [0.01, 1, 100]}  # example: low, medium, high regularization

#logistic regression model
lr_model = LogisticRegression(max_iter=1000)

# setting up GridSearchCV with 5-fold CV
gs_cv = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')

# 8. fiting the model
gs_cv.fit(X_train, y_train)

# 9. printing optimal C and interpretation
best_C = gs_cv.best_params_['C']
print(f"Optimal value for C: {best_C}")

if best_C == min(param_grid['C']):
    print("The optimal C has the most regularization (smallest C).")
elif best_C == max(param_grid['C']):
    print("The optimal C has the least regularization (largest C).")
else:
    print("The optimal C is in the middle of the tested values.")

# test set prediction
df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features])
df_arrests_test['pred_lr_proba'] = gs_cv.predict_proba(df_arrests_test[features])[:, 1]


# save results
df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)
df_arrests_test.to_csv("data/df_arrests_test_with_lr.csv", index=False)




