import pandas as pd
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Import training data
training_df=pd.read_csv(r'C:\Users\astang\OneDrive - Marlabs Inc\Documents\Python Scripts\March_Madness\Training_Data.csv')

#Dictionaries to help with generating dynamic HTML
date_dict={2022: '2022-03-14', 2023: '2023-03-13', 2024: '2024-03-18', 2025: '2025-03-17'}
stat_dict={'OEff': 'offensive-efficiency', 'DEff': 'defensive-efficiency', 'SOS': 'schedule-strength-by-other', 'Turnovers': 'turnovers-per-possession','OReb': 'offensive-rebounding-pct'}

def get_team_rankings(year,stat):
    if stat=='SOS':
        html='https://www.teamrankings.com/ncaa-basketball/ranking/'+stat_dict[stat]+'?date='+date_dict[year]
    else:
        html='https://www.teamrankings.com/ncaa-basketball/stat/'+stat_dict[stat]+'?date='+date_dict[year]
    df=pd.read_html(html)
    return df

def get_yearly_data(year, metric):

    metric_data=get_team_rankings(year,metric)
    metric_data=metric_data[0].iloc[:,[0,1,2]]
    metric_data=metric_data.rename(columns={'Rank': metric+'_Rank'})
    if metric=='SOS':
        metric_data=metric_data.rename(columns={'Rating':metric})
        metric_data['Team'] = metric_data['Team'].str.extract(r'^(.*)\(.*\)', expand=True)
        metric_data['Team'] = metric_data['Team'].str.strip()
    else:
        metric_data=metric_data.rename(columns={str(year-1): metric})

    return metric_data

def merge_metrics(dfs,year):

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Team', how='outer'), dfs)
    merged_df['Year']=year
    return merged_df

data_2022=merge_metrics([get_yearly_data(2022, 'OEff'),get_yearly_data(2022, 'DEff'),get_yearly_data(2022, 'SOS'),get_yearly_data(2022, 'Turnovers'),get_yearly_data(2022, 'OReb')],2022)
data_2023=merge_metrics([get_yearly_data(2023, 'OEff'),get_yearly_data(2023, 'DEff'),get_yearly_data(2023, 'SOS'),get_yearly_data(2023, 'Turnovers'),get_yearly_data(2023, 'OReb')],2023)
data_2024=merge_metrics([get_yearly_data(2024, 'OEff'),get_yearly_data(2024, 'DEff'),get_yearly_data(2024, 'SOS'),get_yearly_data(2024, 'Turnovers'),get_yearly_data(2024, 'OReb')],2024)

#Merge with training data
yearly_data=pd.concat([data_2022,data_2023,data_2024])
yearly_data['Turnovers']=pd.to_numeric(yearly_data['Turnovers'].str.rstrip('%')) / 100
yearly_data['OReb']=pd.to_numeric(yearly_data['OReb'].str.rstrip('%')) / 100
combined_df=pd.merge(training_df,yearly_data,left_on=['Year','Team_1'],right_on=['Year','Team'])
combined_df=pd.merge(combined_df,yearly_data,left_on=['Year','Team_2'],right_on=['Year','Team'])

#Add derived columns
#Used features
def used_features(df):
    df['OEff_Delta']=(df['OEff_x']-df['OEff_y'])
    df['DEff_Delta']=(df['DEff_x']-df['DEff_y'])*-1
    df['ODEff_Delta']=(df['OEff_Delta']+df['DEff_Delta'])*100
    df['SOS_Rank_Delta']=(df['SOS_Rank_x']-df['SOS_Rank_y'])*-1

used_features(combined_df)

#Unused features
combined_df['Seed_Delta']=(combined_df['Team_1_Seed']-combined_df['Team_2_Seed'])*-1
combined_df['SOS_Delta']=combined_df['SOS_x']-combined_df['SOS_y']
combined_df['OEff_Rank_Delta']=(combined_df['OEff_Rank_x']-combined_df['OEff_Rank_y'])*-1/364
combined_df['DEff_Rank_Delta']=(combined_df['DEff_Rank_x']-combined_df['DEff_Rank_y'])*-1/364
combined_df['ODEff_Rank_Delta']=(combined_df['OEff_Rank_Delta']+combined_df['DEff_Rank_Delta'])
combined_df['Turnovers_Delta']=(combined_df['Turnovers_x']-combined_df['Turnovers_y'])*-1
combined_df['OReb_Delta']=(combined_df['OReb_x']-combined_df['OReb_y'])

def classification(row):
    if row['Winner']==row['Team_1']:
        return 1
    else:
        return 0

combined_df['Classification']=combined_df.apply(classification,axis=1)

#Validate output prior to model training
combined_df.to_csv(r'C:\Users\astang\OneDrive - Marlabs Inc\Documents\Python Scripts\March_Madness\test.csv')

#Begin logistic regression
feature_cols=['ODEff_Delta','SOS_Rank_Delta']
X=combined_df[feature_cols] #Features
y=combined_df['Classification'] #Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate view to see features, actual result, predicted result, and implied probabilities
y_pred_df=pd.DataFrame(y_pred,columns=['Prediction'])
probs_df=pd.DataFrame(model.predict_proba(X_test),columns=['Team_2_Win_Prob','Team_1_Win_Prob'])
probs_df=probs_df[['Team_1_Win_Prob','Team_2_Win_Prob']]
sanity_check=pd.concat([X_test.reset_index(drop=True),y_test.reset_index(drop=True),y_pred_df.reset_index(drop=True),probs_df],axis=1)

def predict_it(year,higher_seed,lower_seed):
    #data=get_yearly_data(year)
    data=merge_metrics([get_yearly_data(year, 'OEff'),get_yearly_data(year, 'DEff'),get_yearly_data(year, 'SOS')],year)

    team1_df=data[data['Team']==higher_seed].reset_index(drop=True)
    team2_df=data[data['Team']==lower_seed].reset_index(drop=True)

    #Rename columns
    team1_df=team1_df.rename(columns={'OEff':'OEff_x','DEff':'DEff_x','SOS_Rank':'SOS_Rank_x'})
    team2_df=team2_df.rename(columns={'OEff':'OEff_y','DEff':'DEff_y','SOS_Rank':'SOS_Rank_y'})

    #Join data
    combined_metrics_df=pd.concat([team1_df,team2_df],axis=1)

    #Create derived fields
    used_features(combined_metrics_df)

    #Logistic regression
    matchup=combined_metrics_df[feature_cols]
    y_pred=model.predict(matchup)
    y_pred_df=pd.DataFrame(y_pred,columns=['Prediction'])
    probs_df=pd.DataFrame(model.predict_proba(matchup),columns=['Team_2_Win_Prob','Team_1_Win_Prob'])
    probs_df=probs_df[['Team_1_Win_Prob','Team_2_Win_Prob']]
    pred_df=pd.concat([matchup,y_pred_df,probs_df],axis=1)

    return pred_df

predict_it(2025,"Auburn","Alabama St")