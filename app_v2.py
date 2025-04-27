import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Set page configuration
st.set_page_config(page_title="IPL Match Winner Prediction", layout="centered")

# Initialize session state for toss_winner
if 'toss_winner' not in st.session_state:
    st.session_state.toss_winner = None

# Load the dataset
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/srinathkr07/IPL-Data-Analysis/master/matches.csv"
        df = pd.read_csv(url)
        # Normalize column names: lowercase and strip whitespace
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        return None

# Preprocess data and train model
@st.cache_resource
def train_model(df):
    try:
        # Define features and target
        features = ['team1', 'team2', 'venue', 'toss_winner', 'season', 'dl_applied', 'toss_decision']
        target = 'winner'

        # Verify columns exist
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            st.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Columns {missing_cols} not found in dataset")

        # Select relevant columns and drop rows with missing values
        df = df[features + [target]].dropna()

        # Encode categorical variables
        le_team = LabelEncoder()
        le_venue = LabelEncoder()
        le_toss_winner = LabelEncoder()
        le_winner = LabelEncoder()
        le_toss_decision = LabelEncoder()

        # Combine team1 and team2 for encoding
        all_teams = pd.concat([df['team1'], df['team2']]).unique()
        le_team.fit(all_teams)
        df['team1'] = le_team.transform(df['team1'])
        df['team2'] = le_team.transform(df['team2'])
        df['venue'] = le_venue.fit_transform(df['venue'])
        df['toss_winner'] = le_toss_winner.fit_transform(df['toss_winner'])
        df['winner'] = le_winner.fit_transform(df['winner'])
        df['toss_decision'] = le_toss_decision.fit_transform(df['toss_decision'])

        # Feature Engineering
        df['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)

        # Prepare features and target
        X = df[['team1', 'team2', 'venue', 'toss_winner', 'season', 'dl_applied', 'toss_decision', 'is_toss_winner_team1']]
        y = df['winner']

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        return model, le_team, le_venue, le_toss_winner, le_winner, le_toss_decision, all_teams, le_venue.classes_
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None, None, None

# Predict winner
def predict_winner(model, le_team, le_venue, le_toss_winner, le_winner, le_toss_decision, team1, team2, venue, toss_winner, season, dl_applied, toss_decision):
    try:
        # Encode inputs
        team1_encoded = le_team.transform([team1])[0]
        team2_encoded = le_team.transform([team2])[0]
        venue_encoded = le_venue.transform([venue])[0]
        toss_winner_encoded = le_toss_winner.transform([toss_winner])[0]
        toss_decision_encoded = le_toss_decision.transform([toss_decision])[0]
        is_toss_winner_team1 = 1 if toss_winner == team1 else 0

        # Prepare input data (includes np/array typo)
        input_data = np/array([[team1_encoded, team2_encoded, venue_encoded, toss_winner_encoded, season, dl_applied, toss_decision_encoded, is_toss_winner_team1]])
        
        # Predict
        prediction = model.predict(input_data)
        winner_encoded = prediction[0]
        winner = le_winner.inverse_transform([winner_encoded])[0]
        
        return winner, None
    except ValueError as e:
        return None, f"Error: {str(e)}. Ensure team names, venue, and toss decision are valid."

# Main Streamlit app
def main():
    st.title("IPL Match Winner Prediction")
    st.markdown("Select two teams, venue, toss winner, season, DL method, and toss decision to predict the match winner.")

    # Load data
    df = load_data()
    if df is None:
        return

    # Train model
    model, le_team, le_venue, le_toss_winner, le_winner, le_toss_decision, all_teams, all_venues = train_model(df)
    if model is None:
        return

    # Find indices for default teams
    team1_default = "Mumbai Indians"
    team2_default = "Chennai Super Kings"
    team1_index = list(all_teams).index(team1_default) if team1_default in all_teams else 0
    team2_index = list(all_teams).index(team2_default) if team2_default in all_teams else 1 if len(all_teams) > 1 else 0

    # Create input form
    with st.form(key="prediction_form"):
        team1 = st.selectbox("Team 1", options=all_teams, index=team1_index, key="team1")
        team2 = st.selectbox("Team 2", options=all_teams, index=team2_index, key="team2")
        
        # Update toss_winner options based on team1 and team2 (no robust update)
        toss_winner_options = [team1, team2]
        if st.session_state.toss_winner not in toss_winner_options:
            st.session_state.toss_winner = team1
        toss_winner = st.selectbox(
            "Toss Winner",
            options=toss_winner_options,
            index=toss_winner_options.index(st.session_state.toss_winner) if st.session_state.toss_winner in toss_winner_options else 0,
            key="toss_winner_select"
        )
        st.session_state.toss_winner = toss_winner

        venue = st.selectbox("Venue", options=all_venues, index=0)
        season = st.number_input("Season", min_value=2008, max_value=2019, value=2019)
        dl_applied = st.selectbox("DL Method Applied", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
        toss_decision = st.selectbox("Toss Decision", options=["bat", "field"], index=0)
        submit_button = st.form_submit_button("Predict Winner")

    # Handle form submission
    if submit_button:
        # Validate inputs
        if team1 == team2:
            st.error("Team 1 and Team 2 must be different.")
            return
        if toss_winner not in [team1, team2]:
            st.error("Toss winner must be either Team 1 or Team 2.")
            return

        # Make prediction
        winner, error = predict_winner(
            model, le_team, le_venue, le_toss_winner, le_winner, le_toss_decision,
            team1, team2, venue, toss_winner, season, dl_applied, toss_decision
        )
        if error:
            st.error(error)
        else:
            st.success(f"Predicted Winner: **{winner}**")

if __name__ == "__main__":
    main()