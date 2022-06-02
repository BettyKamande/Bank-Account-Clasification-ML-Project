import category_encoders as ce
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
import xgboost as xgb
import pickle

from xgboost import Booster

st.write('The objective of this app is to create a machine learning model to predict which individuals are most likely '
         'to have or use a bank account.')
st.write("The models and solutions developed can provide an indication of the state of financial inclusion in Kenya,"
         "Rwanda, Tanzania and Uganda,"
         "while providing insights into some of the key factors driving individuals’ financial security.")

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')
train_data = train_data.replace({'location_type': {0: "Rural", 1: "Urban"}})
train_data = train_data.replace({'cellphone_access': {0: "No", 1: "Yes"}})

test_data = test_data.replace({'location_type': {0: "Rural", 1: "Urban"}})
test_data = test_data.replace({'cellphone_access': {0: "No", 1: "Yes"}})

train_data['age_bins'] = pd.cut(train_data['age_of_respondent'], bins=3,
                                labels=['Young', 'Adult', 'Elderly'])
test_data['age_bins'] = pd.cut(train_data['age_of_respondent'], bins=3,
                               labels=['Young', 'Adult', 'Elderly'])
# Title
st.header('Financial Inclusion in Africa')
# Input data
st.date_input("Today's date")
gender_of_respondent = st.selectbox("What is your gender", ("Female", "Male"))
age_of_respondent = st.number_input("Please input your age", min_value=18, max_value=100)
country = st.selectbox("Select country of origin", ("Kenya", "Rwanda", "Tanzania", "Uganda"))
location_type = st.selectbox("Do you live in the rural area or urban area", ("Rural", "Urban"))
cellphone_access = st.selectbox("Do you have access to cellphone", ("Yes", "No"))
relationship_with_head = st.selectbox("What is your relationship with the head of the family", ("Head of Household",
                                                                                                "Spouse", "Child",
                                                                                                "Parent",
                                                                                                "Other relative",
                                                                                                "Other non_relative"))
marital_status = st.selectbox("What is your marital status", ("Married/Living together", "Single/Never Married",
                                                              "Widowed", "Divorced/Separated", "Don't know"))
education_level = st.selectbox("Highest education-level", ("Primary education", "No formal education",
                                                           "Secondary education", "Tertiary education",
                                                           "Vocational/Specialised training", "Other/Don't know/RTA "
                                                           ))
job_type = st.selectbox("Indicate your job-type", ("Self employed", "Informally employed", "Farming and Fishing",
                                                   "Remittance Dependent", "Other Income", "Formally employed Private",
                                                   "No Income", "Formally employed Government", "Government Dependent",
                                                   "Don't Know/Refuse to answer"))
household_size = st.number_input("what is your household-size", min_value=1, max_value=21)

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(train_data)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(all_data, all_data['bank_account'], test_size=0.3,
                                                    random_state=0)

ord_enc = ce.OrdinalEncoder(cols=['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                                  'relationship_with_head', 'marital_status', 'education_level', 'job_type',
                                  'age_bins']).fit(X_train, y_train)

data = ord_enc.transform(all_data)
data.head()

main_cols = data.columns.difference(['bank_account'])

train = data[:ntrain].copy()
# train.drop_duplicates(inplace = True, ignore_index=True)
target = train.bank_account.copy()
train.drop('bank_account', axis=1, inplace=True)

test = data[ntrain:].copy()
test.drop('bank_account', axis=1, inplace=True)
test = test.reset_index(drop=True)

# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("xgboost_model.pkl")
    booster = clf.get_booster()
    booster.save_model('test_model.bin')
    booster: Booster = xgb.Booster()
    booster.load_model('test_model.bin')
    # Store inputs into dataframe

    X = pd.DataFrame(
        [[country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent,
          relationship_with_head, marital_status, education_level, job_type]],
        columns=['country', 'location_type', 'cellphone_access', 'household_siz', 'age_of_respondent',
                 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level',
                 'age_bins', 'job_type'])
    # st.write("From the following data that you have given: ")
    # st.write(X)
    X = X.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = booster.predict(X)[0]
    predict_probability = booster.predict_proba(X)

    # Output prediction

    if prediction == 1:
        st.write(
            f'This respondent is likely to have a bank account. The probability of this respondent having a '
            f'bank account is at  {round(predict_probability[0][1] * 100, 2)}%')
    else:
        st.write(
            f'This respondent is NOT likely to have a bank account. The probability of this respondent NOT having a '
            f'bank account is at  {round(predict_probability[0][0] * 100, 2)}%')
