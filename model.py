import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# check for home ground for any team for feature engineering
HOME_GROUNDS = {
    'Royal Challengers Bengaluru': 'M Chinnaswamy Stadium',
    'Gujarat Titans':              'Narendra Modi Stadium',
    'Sunrisers Hyderabad':         'Rajiv Gandhi International Stadium',
    'Punjab Kings':                'Maharaja Yadavindra Singh',
    'Kolkata Knight Riders':       'Eden Gardens',
    'Mumbai Indians':              'Wankhede Stadium',
    'Lucknow Super Giants':        'BRSABV Ekana Cricket Stadium',
    'Delhi Capitals':              'Arun Jaitley Stadium',
    'Rajasthan Royals':            'Sawai Mansingh Stadium',
    'Chennai Super Kings':         'MA Chidambaram Stadium',
}

df=pd.read_csv("match_data.csv")

# Remvoe duplicates becasue out of two teams both face each other one but here we select two teams 
# so one redundant record need to remove

df.drop_duplicates(subset=['Date',"Team 1","Team 2"],inplace=True)

#in Top_scorer names contain sprcial character that remove for clean data
df['Top_Scorer'] = df['Top_Scorer'].apply(lambda name: str(name).replace('*', '').replace('†', '').strip())


def extract_winner(result):
    """Extracts the winning team name from the result string."""
    for team in HOME_GROUNDS.keys():
        if result.startswith(team):
            return team
    return result.split(' won')[0].strip()

df['winner'] = df['Result'].apply(extract_winner)

# team 1 winner then 1 otherwise 0 means team 2 win
df['target'] = (df['winner'] == df['Team 1']).astype(int)


#converting Date into datetime for feature Extracting
df["Date"]=pd.to_datetime(df["Date"],errors='coerce',dayfirst=True)


#sorting the data on start date to end date for maintain flow of match 
df=df.sort_values("Date").reset_index(drop=True)


#Extracting day of week for model learn hidden patterns that is in weekends matches team won mostly 
df['day_of_week']=df['Date'].dt.dayofweek


# make a new feature is team win on home ground or not 
def is_home(team, venue):
    """Returns 1 if the team is playing at their home ground, else 0."""
    ground = HOME_GROUNDS.get(team, '')
    return 1 if ground and ground.lower() in venue.lower() else 0


df['is_win_team_home_ground'] = df.apply(
    lambda r: is_home(
        r['Team 1'] if r['target'] == 1 else r['Team 2'],
        r['Venue']
    ),
    axis=1
)


# make a score buket of top scorer to learn pattern if top scorer make high runs then it wins or not.
df['score_bucket'] = pd.cut(
    df['Top_Score'],
    bins=[0, 35, 65, 200],
    labels=[0, 1, 2]        
).astype(int)
 



# Encode Team 1 and Team 2 columns as well as venue on lable encoder casue for tree models
encoder=LabelEncoder()
venue_encoder = LabelEncoder()
# Fit on all unique team names first 
all_teams = sorted(set(df['Team 1'].tolist() + df['Team 2'].tolist())) 

# fit on all teams that dataset uniquly identified and then transform

encoder.fit(all_teams) 
df['team1_enc'] = encoder.transform(df['Team 1']) # type: ignore
df['team2_enc'] = encoder.transform(df['Team 2']) # type: ignore


# encode venue column 
df['venue_enc']=venue_encoder.fit_transform(df['Venue']) # type: ignore



# frequency encoding for player names better than label encoding
freq_map = df['Top_Scorer'].value_counts().to_dict()
df['top_scorer_freq'] = df['Top_Scorer'].map(freq_map)
 

#droping unnecessary columns for model training and only keep encoded features and target variable
df.drop(columns=["Date","Team 1","Team 2","Venue","Result","Top_Scorer","Top_Score","winner"],inplace=True)


df.to_csv("transformed_matches.csv",index=False)






# separate independent and dependent variables
X=df.drop(columns=["target"])
y=df["target"]


model=XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# model=LogisticRegression(
#     penalty='l2',          
#     C=1.0,                 
#     solver='lbfgs',        
#     max_iter=100,          
#     random_state=42        
# )

# model = DecisionTreeClassifier(
#     criterion='gini',      
#     max_depth=3,           
#     min_samples_split=2,   
#     min_samples_leaf=1,    
#     random_state=42        
# )




# since dataset is small so we use LeaveOneOut cross validation to evaluate the model performance 
# work on one every time train on 16 point and test on one and repeat all 17 times (dataset sample time)
loo=LeaveOneOut()

# predict the target variable for each test point and evaluate the model performance 
y_pred=cross_val_predict(model,X,y,cv=loo)


# using accuracy, f1 score, classification report and confusion matrix

accuracy=accuracy_score(y,y_pred)

print(f"Model Accuracy: {accuracy:.2f}\n")

f1_Score=f1_score(y,y_pred)
print(f"Model F1 Score: {f1_Score:.2f}\n")

report=classification_report(y,y_pred)
print("Classification Report \n")
print(report)


conf_matrix=confusion_matrix(y,y_pred)
print("Confusion Matrix \n")
print(conf_matrix)






