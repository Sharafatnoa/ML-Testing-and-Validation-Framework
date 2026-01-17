#We will define the columns here for our synthetic data. 

numerical_cols = ['age', 'income', 'debt','savings']
categorical_cols = ['gender','age_group']
target_cols = ['label']


MAX_MISSING_RATE = 0.02
MAX_OUTLIER_RATE = 0.02