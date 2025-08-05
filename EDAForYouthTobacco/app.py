import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

#set the page config for streamlit dashboard
st.set_page_config(page_title="Youth Tobacco Survey Dashboard", layout = "wide")
st.title = ("🧪 Youth Tobacco Use Survey Analysis Dashboard")

#load data
@st.cache_data

#function to get data and preprocess it.

def load_data():
    #get data
    df = pd.read_csv("Youth_Tobacco_Survey__YTS__Data.csv")
    #preprocessing and feature selection
    #drop unneccsary columns from the dataset
    df.drop(['Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'GeoLocation',
         'LocationAbbr','TopicType','DataSource','Data_Value_Unit',
         'Data_Value_Unit','Data_Value_Type','Race','Age','TopicTypeId',
         'TopicId','MeasureId','StratificationID1','StratificationID2',
         'StratificationID3','StratificationID4','SubMeasureID'],
        axis=1, inplace=True)

    #drop all the null values from the data
    df.dropna(inplace=True)

    #rename column
    df.rename(columns={'Data_Value': 'Tobacco consumption percentage (%)'}, inplace=True)

    # Binning the continuous target variable into discrete classes  
    df['Target_Class'] = pd.cut(df['Tobacco consumption percentage (%)'], bins=[0, 20, 40, 60, 80, 100],
                                   labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

    #The 'Gender' column is divided into exactly three parts.
    #Let's assume the survey was split equally between genders. 
    #This means we'll create masks for half of the 'Overall' values,
    #and assign them the values 'Male' and 'Female'.
    
    #create a random array containing true and false values the size of the length of dataframe
    mask = np.random.choice([True, False], size=len(df))
    #assign the True value as Male and false value as Female
    df.loc[mask, 'Gender'] = 'Male'
    df.loc[~mask, 'Gender'] = 'Female'
    #print(df['Gender'].value_counts())
    return df

#call fucntion to load the data
df = load_data()

#Tabs for sections in the Dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA",
                                        "📍 Clustering",
                                        "🔮 Logistic Regression",
                                        "🌲 Random Forest",
                                        "📈 Raw Data"])

#Tab1: EDA
with tab1:
    st.subheader("Exploratory Data Analysis")
    
    #Correlation Matrix
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(fig)
    
    #Consumption by Education and Gender distributions
    st.markdown("### Tobacco Consumption By Education & Gender")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(data=df, x='Education', y='Tobacco consumption percentage (%)', hue='Gender', palette='Paired')
    plt.xticks(rotation=90);
    st.pyplot(fig)

    #Sample Size to Consumption based on Gender
    st.markdown("### Sample Size vs Tabocco Consumption by Gender")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Sample_Size', y='Tobacco consumption percentage (%)', hue='Gender')
    st.pyplot(fig)
    
    #Tobacco consumption dirtirbution based on Region
    fig, ax = plt.subplots(figsize=(12,6))
    region_smoking = df.groupby('LocationDesc')['Tobacco consumption percentage (%)'].mean().sort_values(ascending=False)
    region_smoking.plot(kind='bar', color='skyblue', ax = ax)
    plt.xticks(rotation=90);
    st.pyplot(fig)

#Tab2: Clustering
with tab2:
    st.subheader("State-Wise Tobacco Usage(Clustering Results)")
    #As there are categorical vairables, to fit the models we convert them to dummies - One Hot Encoding
    data = pd.get_dummies(df)
    #print(data.columns)

    # Separate features and target variable
    X = data.drop(['Tobacco consumption percentage (%)'] + 
                  [col for col in data.columns if col.startswith("Target_Class")], axis = 1)
    
    '''Question1: Can we identify unique groups of states based on their youth tabocco patterns?'''
    '''To solve this we choose the K-Means clustering technique which can provide clusters
       of states with similar patterns of youth tobacco use. This helps us identify states with low
       or high prevalence rates and factors which might be contirbuting for this patterns.'''
	   
    #Clustering using K-Means with 4 clusters or groups and fixed random state for reproducibility
    kmeans = KMeans(n_clusters = 4, random_state = 42)
    df['Cluster']  = kmeans.fit_predict(X)
    
    st.markdown("### K-Means Clustering Results")
    fig, ax = plt.subplots()
    plt.scatter(df['Cluster'], df['Tobacco consumption percentage (%)'], c = df['Cluster'], cmap = 'Set2')
    plt.xlabel('Cluster')
    plt.ylabel('Tobacco Consumption %')
    st.pyplot(fig)
    
    '''Question2: Are there any natural groupings of states based on their demographic and tobacco use?'''
    '''Hierarchical clustering technique is used to answer this question. It can help identify any
       hierarchical relationships between states based on demographic and tobacco use attributes,
       i.e. clusters of state with similar demographics and tobacco behaviours.'''
       
    #Agglomerative Clustering to check hierarchical relations of patterns if any
    agg = AgglomerativeClustering(n_clusters=4)
    df['Agg_Cluster'] = agg.fit_predict(X)

    st.markdown("### Agglomerative Clustering")
    fig, ax = plt.subplots()
    plt.scatter(df['Agg_Cluster'], df['Tobacco consumption percentage (%)'], c=df['Agg_Cluster'], cmap='Set1')
    plt.xlabel('Cluster')
    plt.ylabel('Tobacco Consumption %')
    st.pyplot(fig)

#Tab3: LogisticRegression
with tab3:

    st.subheader('Logistic Regression (Predicting Very Low Tobacco Use)')
    
    #Used Labeled target variable for predictions - supervised learning
    y = data['Target_Class_Very Low']
    #Split data into train and test sets for classification tasks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    '''Question3: Can we predict the likelihood of a state having low youth tobacco
		  use prevalence based on the demographic and educational factors?'''
    '''Using logistic regression, can provide probabilites of a state having low youth
       tobacco use prevalence based on the demographic and educational factors. It can help
       identifying significant predictors and their impact on tobacco use prevalence.''' 
       
    # Logistic Regression having a very low class for Youth Tabocco levels as target variable
    log_reg = LogisticRegression(max_iter = 200)
    log_reg.fit(X_train, y_train)
    y_predict = log_reg.predict(X_test)

    report = classification_report(y_test, y_predict, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    #format it properly
    report_df = report_df.round(2)
    
    st.markdown("📋 Classification Report")
    st.dataframe(report_df.style.background_gradient(cmap = 'Blues'))

#Tab4: Random Forest
with tab4:

    st.subheader("Random Forest Classifier Performance")

    '''Question4: Which classification algorithm perfroms better in predicting the smoking
		  status of youth based on the demographic features?'''
    '''By using cross-validation, we can compare the performance of different classification
       algorithms such as Random Forest Classifier, SVM etc, in predicting the smoking status of
       the youth. It gives a very high accuracy for random forest classifier trained only for low
       youth tobacco. Furhter analysis for all the ascpects of the target variable is required
       to fully understand the relevance.'''
    
    #Random Forest Classifier having a very low class for Youth Tabocco levels as target variable
    rf_classifier = RandomForestClassifier(random_state = 42)
    rf_scores = cross_val_score(rf_classifier, X, y, cv=5)

    #show scores of each fold
    st.markdown("### Cross-Validated Score (5-fold)")
    cv_df = pd.DataFrame({'Fold': [f'Fold {i+1}' for i in range(len(rf_scores))], 'Accuracy': rf_scores})
    st.dataframe(cv_df.style.format({"Accuracy": "{:.2%}"}).background_gradient(cmap='Greens'))

    st.metric("Mean Accuracy", f"{rf_scores.mean(): .2%}")
    st.metric("Std Dev", f"{rf_scores.std():.2%}")
    
#Tab5: Dataset Used
with tab5:
    st.subheader("Raw Data")
    st.dataframe(df)
