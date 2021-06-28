import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@st.cache()
def load_data():
    file_path = 'https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/glass-types.csv'
    df = pd.read_csv(file_path, header = None)
    
    # Drop the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
        # A Python list containing the suitable column headers as string values. Also, create a Python dictionary as described above.
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
        # Required Python dictionary.
    columns_dict = {}
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
            # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)

    return df

glass_df = load_data() 
# Creating the features data-frame holding all the columns except the last column
x = glass_df.iloc[:, :-1]


# Creating the target series that holds last column 
y = glass_df['GlassType']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# Create a function 'prediction()' which accepts SepalLength, SepalWidth, PetalLength, PetalWidth as input and returns species name.
@st.cache()
def prediction(model,ri, na, mg, al, si, k, ca, ba, fe):
    type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    type = type[0]
    if type == 1:
        return "building_windows_float_processed"
    elif type == 2:
        return "building_windows_non_float_processed"
    elif type == 3:
        return "vehicle_windows_float_processed"
    elif type == 4:
        return "vehicle_windows_non_float_processed"
    elif type == 5:
        return "containers" 
    elif type == 6:
        return "tableware"
    else:
        return "headlamps"

st.title("Glass Type prediction Web app")
st.sidebar.title("Glass Type prediction Web app")

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Glass Type Data set")
    st.dataframe(glass_df)
    

import seaborn as sns
import matplotlib.pyplot as plt


st.sidebar.subheader("Visualisation Selector")
plot_list = st.sidebar.multiselect("Select the Charts/Plots:",('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))

st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Correlation Heatmap' in plot_list:
	st.subheader("Correlation Heatmap")
	sns.heatmap(df.corr(),annot=True)
	st.pyplot()

if 'Line Chart' in plot_list:
	st.subheader("Line Chart")
	st.line_chart(glass_df)
			
if 'Area Chart' in plot_list:
	st.subheader("Area Chart")
	st.area_chart(glass_df)

if 'Count Plot' in plot_list:
	st.subheader("Count plot")
	sns.countplot('GlassType',data=glass_df)
	st.pyplot()

if 'Pie Chart' in plot_list:
	st.subheader("Pie Chart")
	pie_data = glass_df['GlassType'].value_counts()
	plt.pie(pie_data, labels=pie_data.index, autopct='%1.2f%%', startangle=30)
	st.pyplot()

if 'Box Plot' in plot_list:
	st.subheader("Box Plot")
	variable = st.sidebar.selectbox("Select the variable for boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
	sns.boxplot(glass_df[variable])
	st.pyplot()


st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri",float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na",float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg",float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al",float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si",float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K",float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca",float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba",float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe",float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier'))

from sklearn.metrics import plot_confusion_matrix

if classifier=='Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_input = st. sidebar.number_input("C (Error Rate)")
    kernel_input = st.sidebar.radio("Kernel",("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma")

    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C=c_input, kernel=kernel_input, gamma=gamma_input)
        svc_model.fit(x_train,y_train)
        y_pred = svc_model.predict(x_test)
        accuracy = svc_model.score(x_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, x_test, y_test)
        st.pyplot()
                      
            


if classifier =='Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest",100, 5000,step=10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1)
        


    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf= RandomForestClassifier(n_estimators=n_estimators_input, max_depth=max_depth_input, n_jobs=-1)
        rf_clf.fit(x_train,y_train)
        accuracy = rf_clf.score(x_test, y_test)
        glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rf_clf, x_test, y_test)
        st.pyplot()
