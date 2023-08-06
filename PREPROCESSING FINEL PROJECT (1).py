#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns


# In[2]:


import pandas as pd
import io
import requests

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
response = requests.get(URL)

if response.status_code == 200:
    df = pd.read_csv(io.BytesIO(response.content))
    print(df.head(5))
else:
    print("Failed to fetch data from the URL.")


# In[3]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the SpaceX dataset into a Pandas dataframe
#df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# EDA - FlightNumber vs. PayloadMass vs. Launch Outcome
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Payload Mass (kg)", fontsize=20)
plt.show()



# In[4]:


#EDA - Relationship between FlightNumber and Launch Site
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, kind="swarm")
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Launch Site", fontsize=20)
plt.show()


# In[5]:


# EDA - Relationship between PayloadMass and Launch Site
sns.catplot(x="PayloadMass", y="LaunchSite", hue="Class", data=df, kind="swarm")
plt.xlabel("Payload Mass (kg)", fontsize=20)
plt.ylabel("Launch Site", fontsize=20)
plt.show()


# In[6]:


#EDA - Relationship between Success Rate and Orbit Type
orbit_success_rate = df.groupby("Orbit")["Class"].mean().reset_index()
sns.barplot(x="Class", y="Orbit", data=orbit_success_rate)
plt.xlabel("Success Rate", fontsize=20)
plt.ylabel("Orbit", fontsize=20)
plt.show()


# In[7]:


# EDA - Relationship between FlightNumber and Orbit Type
sns.catplot(x="FlightNumber", y="Orbit", hue="Class", data=df, kind="swarm")
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Orbit", fontsize=20)
plt.show()


# In[8]:


# EDA - Relationship between PayloadMass and Orbit Type
sns.catplot(x="PayloadMass", y="Orbit", hue="Class", data=df, kind="swarm")
plt.xlabel("Payload Mass (kg)", fontsize=20)
plt.ylabel("Orbit", fontsize=20)
plt.show()


# In[9]:


# EDA - Launch Success Yearly Trend
df['Year'] = pd.to_datetime(df['Date']).dt.year
yearly_success_rate = df.groupby("Year")["Class"].mean().reset_index()
sns.lineplot(x="Year", y="Class", data=yearly_success_rate)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Average Success Rate", fontsize=20)
plt.show()


# In[ ]:





# In[ ]:





# In[10]:


# Selecting features for prediction
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

# Create dummy variables for categorical columns
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])

# Cast all numeric columns to float64
features_one_hot = features_one_hot.astype('float64')

# Display the resulting dataframe
print(features_one_hot.dtypes)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:





# In[ ]:





# In[ ]:





# In[12]:


get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite://')



# In[13]:


import sqlite3


# In[59]:


conn = sqlite3.connect('mydatabase.db')

# File names and corresponding table names
files = {
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv": "SPACEXTBL"
}

# Load the CSV data into pandas dataframes and insert them into the database
for file, table in files.items():
    df = pd.read_csv(file)
    df.to_sql(table, conn, if_exists='replace', index=False)

# Use the %sql magic command to run SQL queries
get_ipython().run_line_magic('sql', 'sqlite:///mydatabase.db')

# Task 1: Display the names of the unique launch sites in the space mission
get_ipython().run_line_magic('sql', 'SELECT DISTINCT "Launch_Site" FROM SPACEXTBL;')



# In[60]:


# Task 2: Display 5 records where launch sites begin with the string 'CCA'
get_ipython().run_line_magic('sql', 'SELECT * FROM SPACEXTBL WHERE "Launch_Site" LIKE \'CCA%\' LIMIT 5;')



# In[61]:


# Task 3: Display the total payload mass carried by boosters launched by NASA (CRS)
get_ipython().run_line_magic('sql', 'SELECT SUM("PAYLOAD_MASS__KG_") AS TotalPayloadMass FROM SPACEXTBL WHERE "Launch_Site" = \'CCAFS SLC-40\';')


# In[62]:


# Task 4: Display average payload mass carried by booster version F9 v1.1
get_ipython().run_line_magic('sql', 'SELECT AVG("PAYLOAD_MASS__KG_") AS avg_payload_mass FROM SPACEXTBL WHERE "Booster_Version" = \'F9 v1.1\';')




# In[63]:


# Task 5: List the date when the first successful landing outcome on the ground pad was achieved
get_ipython().run_line_magic('sql', 'SELECT MIN("Date") AS first_successful_ground_landing FROM SPACEXTBL WHERE "Landing_Outcome" = \'Success (ground pad)\';')


# In[64]:


get_ipython().run_cell_magic('sql', 'SELECT DISTINCT "Booster_Version"', 'FROM SPACEXTBL\nWHERE "Landing_Outcome" = \'Success (drone ship)\' \n  AND "PAYLOAD_MASS__KG_" > 4000 \n  AND "PAYLOAD_MASS__KG_" < 6000;\n\n\n\n')


# In[65]:


# Task 7: List the total number of successful and failure mission outcomes
get_ipython().run_line_magic('sql', 'SELECT "Mission_Outcome", COUNT(*) AS "Total_Count" FROM SPACEXTBL GROUP BY "Mission_Outcome";')


# In[66]:


get_ipython().run_cell_magic('sql', 'SELECT "Booster_Version"', 'FROM SPACEXTBL\nWHERE "PAYLOAD_MASS__KG_" = (\n    SELECT MAX("PAYLOAD_MASS__KG_")\n    FROM SPACEXTBL\n);\n')


# In[76]:


get_ipython().run_cell_magic('sql', '', 'SELECT\n    CASE substr("Date", 4, 2)\n        WHEN \'01\' THEN \'January\'\n        WHEN \'02\' THEN \'February\'\n        WHEN \'03\' THEN \'March\'\n        WHEN \'04\' THEN \'April\'\n        WHEN \'05\' THEN \'May\'\n        WHEN \'06\' THEN \'June\'\n        WHEN \'07\' THEN \'July\'\n        WHEN \'08\' THEN \'August\'\n        WHEN \'09\' THEN \'September\'\n        WHEN \'10\' THEN \'October\'\n        WHEN \'11\' THEN \'November\'\n        WHEN \'12\' THEN \'December\'\n    END AS Month,\n    "Landing_Outcome",\n    "Booster_Version",\n    "Launch_Site" \nFROM SPACEXTBL;\n')


# In[73]:


import pandas as pd

query = """
SELECT 
    CASE substr("Date", 4, 2)
        WHEN '01' THEN 'January'
        WHEN '02' THEN 'February'
        WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'
        WHEN '05' THEN 'May'
        WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'
        WHEN '08' THEN 'August'
        WHEN '09' THEN 'September'
        WHEN '10' THEN 'October'
        WHEN '11' THEN 'November'
        WHEN '12' THEN 'December'
    END AS Month,
    "Landing_Outcome",
    "Booster_Version",
    "Launch_Site"
FROM SPACEXTBL
WHERE substr("Date", 7, 4) = '2015'
  AND "Landing_Outcome" LIKE '%Failure (drone ship)%';
"""

result = get_ipython().run_line_magic('sql', '$query')

# Convert the SQL query result to a pandas DataFrame
df_result = result.DataFrame()

# Display the DataFrame
display(df_result)


# In[74]:


print(result)


# In[69]:


get_ipython().run_cell_magic('sql', '', 'SELECT "Landing_Outcome", COUNT(*) AS "Count"\nFROM SPACEXTBL\nWHERE "Date" >= \'2010-06-04\' AND "Date" <= \'2017-03-20\'\nGROUP BY "Landing_Outcome"\nORDER BY "Count" DESC;\n')


# In[77]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


# In[80]:


print(df)


# In[78]:


app = dash.Dash(__name__)

app.layout = html.Div([
    # Task 1: Add a Launch Site Drop-down Input Component
    # Task 2: Add a callback function to render success-pie-chart based on selected site dropdown

    # Task 3: Add a Range Slider to Select Payload

    # Task 4: Add a callback function to render the success-payload-scatter-chart scatter plot
])


# In[88]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load your SpaceX launch dataset (replace 'your_dataset.csv' with the actual file path):

# Assuming you have a 'Launch Site' column in your DataFrame
launch_sites = df['Launch_Site'].unique()
dropdown_options = [{'label': site, 'value': site} for site in launch_sites]

# Create the Dash app
app = dash.Dash(__name__)

# Task 1 & 2: Add a Launch Site Drop-down Input Component and corresponding callback function
@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('launch-site-dropdown', 'value')
)
def update_pie_chart(selected_site):
    filtered_df = df[df['Launch_Site'] == selected_site]
    success_counts = filtered_df['Mission_Outcome'].value_counts()
    fig = px.pie(success_counts, values='Mission_Outcome', names=success_counts.index)
    return fig

# Task 3 & 4: Add a Range Slider to Select Payload and corresponding callback function
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    Input('payload-range-slider', 'value')
)
def update_scatter_chart(selected_payload_range):
    filtered_df = df[(df['PAYLOAD_MASS__KG_'] >= selected_payload_range[0]) & (df['PAYLOAD_MASS__KG_'] <= selected_payload_range[1])]
    fig = px.scatter(filtered_df, x='PAYLOAD_MASS__KG_', y='Mission_Outcome', color='Booster_Version')
    return fig

# Layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='launch-site-dropdown',
        options=dropdown_options,
        value=launch_sites[0]
    ),
    dcc.Graph(id='success-pie-chart'),
    dcc.RangeSlider(
        id='payload-range-slider',
        min=df['PAYLOAD_MASS__KG_'].min(),
        max=df['PAYLOAD_MASS__KG_'].max(),
        value=[df['PAYLOAD_MASS__KG_'].min(), df['PAYLOAD_MASS__KG_'].max()],
        marks={str(mass): str(mass) for mass in df['PAYLOAD_MASS__KG_'].unique()}
    ),
    dcc.Graph(id='success-payload-scatter-chart')
    # Other components for other tasks can be added here
])

if __name__ == '__main__':
    app.run_server(debug=True)



# In[87]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the dataframe
data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
data = pd.read_csv(data_url)

# Task 1: Create Y array
Y = data["Class"].to_numpy()
Y = pd.Series(Y)

# Task 2: Standardize the data
X_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
X = pd.read_csv(X_url)
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# Task 3: Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Task 4: Logistic Regression and GridSearchCV
parameters = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)

# Task 5: Calculate accuracy on test data
accuracy_lr = logreg_cv.score(X_test, Y_test)
print("Logistic Regression Accuracy:", accuracy_lr)

# Task 6: Confusion Matrix for Logistic Regression
yhat_lr = logreg_cv.predict(X_test)
cm_lr = confusion_matrix(Y_test, yhat_lr)
plt.figure()
sns.heatmap(cm_lr, annot=True, cmap='Blues', fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




