#CSCI 183
#Final Project
#Jack Corley, Savannah Balistreri, and Lauren Vu 
#FOR PREPARING DATA TO MAKE NEW MODELS WITHOUT RACE AND GENDER

import pandas as pd

df = pd.read_csv('resume.csv')  

feature_to_remove = 'race'  

if feature_to_remove in df.columns:
    df.drop(feature_to_remove, axis=1, inplace=True)
    df.to_csv('removed_resume.csv', index=False)  
else:
    print(f"The feature '{feature_to_remove}' is not found in the DataFrame.")

feature_to_remove = 'gender' 

if feature_to_remove in df.columns:
    df.drop(feature_to_remove, axis=1, inplace=True)
    df.to_csv('removed_resume.csv', index=False)  
else:
    print(f"The feature '{feature_to_remove}' is not found in the DataFrame.")