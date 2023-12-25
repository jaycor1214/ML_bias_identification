#CSCI 183
#Final Project
#Jack Corley, Savannah Balistreri, and Lauren Vu 
# FOR CHANGING RACE AND GENDER CATEGORIES 

import pandas as pd

df = pd.read_csv('resume.csv')

# df_all_men = df.copy()
# df_all_women = df.copy()
# df_all_black = df.copy()
# df_all_white = df.copy()
df_all_black_women = df.copy()

# df_all_men['gender'] = 'm'
# df_all_women['gender'] = 'f'
# df_all_black['race'] = 'black'
# df_all_white['race'] = 'white'
df_all_black_women['race'] = 'white'
df_all_black_women['gender'] = 'm'

# df_all_men.to_csv('all_men_resume.csv', index=False)
# df_all_women.to_csv('all_women_resume.csv', index=False)
# df_all_black.to_csv('all_black_resume.csv', index=False)
# df_all_white.to_csv('all_white_resume.csv', index=False)
df_all_black_women.to_csv('all_white_men_resume', index=False)

