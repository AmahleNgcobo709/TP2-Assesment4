import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time # To simulate model training

#Page Configuration
st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching the data loading function for performance
@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the health data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Basic preprocessing
        # Fill missing 'gender' and 'education' with the mode (most frequent value)
        if 'gender' in df.columns and df['gender'].isnull().any():
            df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
        if 'education' in df.columns and df['education'].isnull().any():
            df['education'] = df['education'].fillna(df['education'].mode()[0])
        
        # FIX: Fill missing 'chronic_illness' values to prevent Plotly error
        if 'chronic_illness' in df.columns and df['chronic_illness'].isnull().any():
            df['chronic_illness'] = df['chronic_illness'].fillna('Not Specified')
            
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the script.")
        return None

#Load the data
df = load_data('merged_cleaned.csv')

#Machine Learning Model Training
@st.cache_resource
def train_model(data):
    """Trains a RandomForestClassifier model."""
    df_model = data.copy()
    
    # Define features (X) and target (y)
    # Using 'chronic_flag' as the target variable (1 for chronic illness, 0 for none)
    features = ['age', 'gender', 'education', 'income', 'gp_visits', 'province']
    target = 'chronic_flag'
    
    X = df_model[features]
    y = df_model[target]
    
    # One-Hot Encode categorical features
    X_encoded = pd.get_dummies(X, columns=['gender', 'education', 'province'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the trained model, encoded columns, and accuracy
    return model, X_encoded.columns, accuracy

# Check if data is loaded before training
if df is not None:
    # Train the model and get necessary components
    model, model_columns, accuracy = train_model(df)
else:
    # Stop the app if data couldn't be loaded
    st.stop()


#Sidebar for User Inputs
st.sidebar.header("👤 Patient Profile for Prediction")
st.sidebar.write("Adjust the sliders and dropdowns to get a prediction.")

# Input fields in the sidebar
age = st.sidebar.slider("Age", 18, 100, 45)
gp_visits = st.sidebar.slider("Annual GP Visits", 0, 10, 3)
income = st.sidebar.slider("Household Income (ZAR)", 0, 50000, 25000)

gender = st.sidebar.selectbox("Gender", df['gender'].unique())
education = st.sidebar.selectbox("Education Level", df['education'].unique())
province = st.sidebar.selectbox("Province", df['province'].unique())


#Main Page Layout
st.title("🩺 Health Analytics & Chronic Illness Prediction")
st.write("An interactive dashboard to explore health data and predict chronic illness likelihood.")

#Prediction Logic and Display
# Create a button to trigger prediction
if st.sidebar.button("🔮 Predict Chronic Illness", use_container_width=True):
    # Create a DataFrame from user inputs
    input_data = {
        'age': [age],
        'gender': [gender],
        'education': [education],
        'income': [income],
        'gp_visits': [gp_visits],
        'province': [province]
    }
    input_df = pd.DataFrame(input_data)

    # One-Hot Encode the input DataFrame
    input_encoded = pd.get_dummies(input_df, columns=['gender', 'education', 'province'])
    
    # Align columns with the model's training columns
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    with st.spinner('Analyzing profile...'):
        time.sleep(1) # Simulate processing
        prediction = model.predict(input_aligned)[0]
        prediction_proba = model.predict_proba(input_aligned)[0]

    # Display prediction result in the main area
    st.subheader("Prediction Result")
    if prediction == 1:
        probability_percent = prediction_proba[1] * 100
        st.error(f"**High Risk of Chronic Illness** (Probability: {probability_percent:.2f}%)", icon="💔")
        st.write("Based on the provided profile, the model predicts a high likelihood of having a chronic illness. Please consult a healthcare professional for advice.")
    else:
        probability_percent = prediction_proba[0] * 100
        st.success(f"**Low Risk of Chronic Illness** (Probability: {probability_percent:.2f}%)", icon="💚")
        st.write("The model predicts a low likelihood of a chronic illness based on this profile. Continue to maintain a healthy lifestyle.")
    
    st.info(f"Model Accuracy: {accuracy:.2%}", icon="🎯")

st.markdown("---")


#Data Visualization Section
st.header("📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
    fig_age = px.histogram(df, x='age', nbins=30, title="Distribution of Patient Age", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("Chronic Illness by Gender")
    fig_gender = px.sunburst(
        df, 
        path=['gender', 'chronic_illness'], 
        title="Chronic Illness Breakdown by Gender",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_gender, use_container_width=True)

st.subheader("Health Expenditure vs. Income by Province")
fig_scatter = px.scatter(
    df, 
    x='income', 
    y='health_expenditure', 
    color='province',
    size='household_size',
    hover_name='province',
    title="Income vs. Health Expenditure Across Provinces",
    log_x=True, # Use logarithmic scale for better visualization of income spread
    size_max=20,
)
st.plotly_chart(fig_scatter, use_container_width=True)


#Raw Data Section
st.markdown("---")
st.header("📄 Explore the Dataset")
if st.checkbox("Show Raw Data", value=False):
    st.write(f"Displaying the first 100 rows of {len(df)} total records.")
    st.dataframe(df.head(100))
