import pandas as pd
import numpy as np

# 1. Load the datasets
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

# %% [2] Merging and Cleaning
# The column in matches.csv is 'match_id', not 'id'
df = deliveries.merge(matches[['match_id', 'venue', 'team1', 'team2', 'winner']], 
                      on='match_id')

# Note: We use 'on' instead of 'left_on/right_on' because the name 
# is now the same in both files.

# 3. Clean Team Names (Handling 2025 updates)
# Standardizing 'Bangalore' to 'Bengaluru' and other variations
def standardize_teams(team):
    if 'Bangalore' in team or 'Bengaluru' in team: return 'Royal Challengers Bengaluru'
    if 'Kings XI Punjab' in team: return 'Punjab Kings'
    if 'Delhi Daredevils' in team: return 'Delhi Capitals'
    return team

df['batting_team'] = df['batting_team'].apply(standardize_teams)
df['bowling_team'] = df['bowling_team'].apply(standardize_teams)

# 4. Feature Engineering: Current State
# Calculate current score and wickets fallen per match per innings
df['current_score'] = df.groupby(['match_id', 'innings'])['total_runs'].cumsum()
df['wickets_fallen'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()

# Convert 'over' and 'ball' into a single 'balls_bowled' feature
df['balls_bowled'] = (df['over'] * 6) + df['ball']
df['balls_left'] = 120 - df['balls_bowled']

# 5. Feature Engineering: Rolling Momentum (Last 5 Overs)
# This is the "secret sauce" for professional sports models
df['runs_last_5'] = (df.groupby(['match_id', 'innings'])['total_runs']
                     .rolling(window=30).sum()
                     .reset_index(level=[0,1], drop=True))

df['wickets_last_5'] = (df.groupby(['match_id', 'innings'])['is_wicket']
                        .rolling(window=30).sum()
                        .reset_index(level=[0,1], drop=True))

# 6. Define the Target Variable
# For the Score Predictor, we want to know the TOTAL runs scored in that innings
total_score_df = df.groupby(['match_id', 'innings'])['total_runs'].sum().reset_index()
total_score_df.rename(columns={'total_runs': 'final_score'}, inplace=True)
df = df.merge(total_score_df, on=['match_id', 'innings'])

# 7. Final Cleanup
# We only want data after 5 overs (30 balls) so the 'runs_last_5' column is full
final_df = df[df['balls_bowled'] >= 30]

# Keep only the columns needed for ML
relevant_cols = [
    'batting_team', 'bowling_team', 'venue', 'current_score', 
    'balls_left', 'wickets_fallen', 'runs_last_5', 'wickets_last_5', 'final_score'
]
final_df = final_df[relevant_cols]

# Save the cleaned data for the next step (Training)
final_df.to_csv('ipl_ml_ready.csv', index=False)
print("Success! 'ipl_ml_ready.csv' created with rolling features.")