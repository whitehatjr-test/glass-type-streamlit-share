import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import confusion_matrix, classification_report

st.set_option('deprecation.showPyplotGlobalUse', False)
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

df = load_data() 
# Creating the features data-frame holding all the columns except the last column
x = df.iloc[:, :-1]


# Creating the target series that holds last column 
y = df['GlassType']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# Create a function 'prediction()' which accepts SepalLength, SepalWidth, PetalLength, PetalWidth as input and returns species name.
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
    st.dataframe(df)
    

import seaborn as sns
import matplotlib.pyplot as plt


st.sidebar.subheader("Exploratory Data Analysis")
chart_type = st.sidebar.multiselect("Visualisation Type",('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))
if 'Correlation Heatmap' in chart_type:
	st.subheader("Correlation Heatmap")
	sns.heatmap(df.corr(),annot=True)
	st.pyplot()

if 'Line Chart' in chart_type:
	st.subheader("Line Chart")
	st.line_chart(df)
			
if 'Area Chart' in chart_type:
	st.subheader("Area Chart")
	st.area_chart(df)

if 'Count Plot' in chart_type:
	st.subheader("Count plot")
	sns.countplot('GlassType',data=df)
	st.pyplot()

if 'Pie Chart' in chart_type:
	st.subheader("Pie Chart")
	pie_data = df['GlassType'].value_counts()
	explode = np.linspace(0, 0.5, 7) # Shift the slices away from the centre of the pie 
	plt.pie(pie_data, labels=pie_data.index, autopct='%1.2f%%', startangle=30)
	st.pyplot()

if 'Box Plot' in chart_type:
	st.subheader("Box Plot")
	variable = st.sidebar.selectbox("Select the variable for boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
	sns.boxplot(df[variable])
	st.pyplot()


st.sidebar.subheader("Select your values:")
Ri = st.sidebar.slider("Input Ri",0.0,100.0)
Na = st.sidebar.slider("Input Na",0.0,100.0)
Mg = st.sidebar.slider("Input Mg",0.0,100.0)
Al = st.sidebar.slider("Input Al",0.0,100.0)
Si = st.sidebar.slider("Input Si",0.0,100.0)
K = st.sidebar.slider("Input K",0.0,100.0)
Ca = st.sidebar.slider("Input Ca",0.0,100.0)
Ba = st.sidebar.slider("Input Ba",0.0,100.0)
Fe = st.sidebar.slider("Input Fe",0.0,100.0)

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier'))


if classifier=='Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    C = st. sidebar.number_input("C (Reglarisation paramter)",0.01, 10.0, step=0.01)
    kernel = st.sidebar.radio("Kernel",("rbf","linear"))
    gamma = st.sidebar.radio("Gamma (Kernel coefficicent)", ("scale","auto"))

    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C=C, kernel=kernel, gamma=gamma)
        svc_model.fit(x_train,y_train)
        y_pred = svc_model.predict(x_test)
        accuracy = svc_model.score(x_test, y_test)
        glass_type = prediction(svc_model, Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, x_test, y_test)
        st.pyplot()
                      
            


if classifier=='Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("Number of trees in the forest",100, 5000,step=10)
    max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 20, step=1)
        


    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rfc_model= RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        rfc_model.fit(x_train,y_train)
        y_pred = rfc_model.predict(x_test)
        accuracy = rfc_model.score(x_test, y_test)
        glass_type = prediction(rfc_model, Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rfc_model, x_test, y_test)
        st.pyplot()