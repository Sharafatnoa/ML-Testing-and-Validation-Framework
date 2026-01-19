#We will make a helper function to generate our test datas here.
import numpy as np
import pandas as pd

def makeDataSet(n=2000, seed=42, missing_rate=0.00) -> pd.DataFrame:
    rng = np.random.default_rng(seed) #This creates a modern numpy random generator object.

    age = rng.integers(18, 60, size=n) #Pick random integers between 18 and 59 total n rows
    gender = rng.choice(['F', 'M'], size=n)

    #Income
    # generate values around mean = 35,000
    # with spread (std = 12,000)
    # then .clip() prevents unrealistic values:
            # income won’t be below 5,000
            # income won’t exceed 150,000
    income = rng.normal(35000, 12000, size=n).clip(5000,150000)
    debt = rng.normal(8000, 5000, size=n).clip(0, 60000)
    savings = rng.normal(6000, 7000, size=n).clip(0, 80000)

    # age_group
    # This takes age values and categorizes them:
        # 18–25 → "18-25"
        # 26–40 → "26-40"
        # 41–60 → "41-60"
    # Example:
        # age = 23 → age_group = "18-25"
        # age = 34 → "26-40"
        # age = 52 → "41-60"
    age_group = pd.cut(age, bins=[17,25,40,60], labels=["18-25","26-40","41-60"])

   
    #Compute the Label

    #Task A: Create a risk score
    score = 0.00005 * income - 0.00008 * debt + 0.00003 * savings
    # This is a simple “decision formula”:
    # income increases score (positive sign)
    # debt reduces score (negative sign)
    # savings increases score (positive sign)
    # Think of score like: “How likely this person is to be label=1”

    #Task B: Convert Score to Probability
    P = 1/(1+np.exp(-score))
    # This is the sigmoid/logistic function.
    # It converts any number into a probability between 0 and 1:
    # big score → p closer to 1
    # small/negative score → p closer to 0
    # Example:
    # score = 2 → p ≈ 0.88
    # score = -2 → p ≈ 0.12

    #Task C: Create The Final Label 0/1
    y = (rng.random(n)<P).astype(int)
    # If random < probability → label becomes 1
    # Else → label becomes 0

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "age_group": age_group,
        "income": income,
        "debt": debt,
        "savings": savings,
        "label": y
    })
    #Inject missing values
    if missing_rate > 0:
        m = int(n * missing_rate)
        idx = rng.choice(df.index, size=m, replace=False)
        df.loc[idx, "savings"] = np.nan
    return df