#We will define the columns here for our synthetic data. 

numerical_cols = ['age', 'income', 'debt','savings']
categorical_cols = ['gender','age_group']
target_cols = 'label'

All_Cols = numerical_cols + categorical_cols + [target_cols]

#-------allowed values/Ranges---------
age_min = 18
age_max = 60

gender_values = ["F", "M"]
age_group_values = ["18-25","26-40","41-60"]
label_values = [0,1]

#-----------------Quality Gates------------------
MAX_MISSING_RATE = 0.02
MAX_OUTLIER_RATE = 0.02