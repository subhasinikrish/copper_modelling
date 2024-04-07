import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.model_selection import GridSearchCV
import streamlit as st
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#download file
df=pd.read_excel(r"C:\Users\kbrad\Downloads\Copper_Set.xlsx")

# changing wrong datatypes specially for numeric values
df['item_date']=pd.to_datetime(df['item_date'],format='%Y%m%d',errors='coerce').dt.date
df['quantity tons']=pd.to_numeric(df['quantity tons'],errors='coerce')
df['material_ref']=df['material_ref'].str.lstrip("0")
df['delivery date']=pd.to_datetime(df['delivery date'],format='%Y%m%d',errors='coerce').dt.date

# clearning null values using mode,mean and median functions
#df.isna().sum()
df.dropna(subset=['id'],inplace=True)
df['item_date']=df['item_date'].fillna(df['item_date'].mode()[0])
df['quantity tons']=df['quantity tons'].fillna(df['quantity tons'].mean())
df['customer']=df['customer'].fillna(df['customer'].median())
df['country']=df['country'].fillna(df['country'].mean())
df['status']=df['status'].fillna(df['status'].mode()[0])
df['application']=df['application'].fillna(df['application'].median())
df['thickness']=df['thickness'].fillna(df['thickness'].mean())
df['material_ref']=df['material_ref'].fillna(df['material_ref'].mode()[0])
df['delivery date']=df['delivery date'].fillna(df['delivery date'].mode()[0])
df['selling_price']=df['selling_price'].fillna(df['selling_price'].median())

#coping file in another name for modifications

df_p=df.copy()

#skewness and eradicate outliers

import seaborn as sns
import numpy as np
df_p['selling_price_log']=np.log(df_p['selling_price'])

df_p['quantity tons_log']=np.log(df_p['quantity tons'])

df_p['thickness_log']=np.log(df_p['thickness'])


mask1=df_p['selling_price']<=0
#print(mask1.sum())
df_p.loc[mask1,'selling_price']=np.nan
mask2=df_p['quantity tons']<=0
#print(mask2.sum())
df_p.loc[mask2,'quantity tons']=np.nan
mask2=df_p['thickness']<=0
#print(mask2.sum())

#df_p.isna().sum()
df_p=df_p.dropna()
#df_p.isna().sum()







st.set_page_config(layout='wide')
st.title(":green[INDUSTRIAL COPPER MODELING PROJECT]")

tab1,tab2=st.tabs(["PREDICT SELLING PRICE","PREDICT STATUS"])

with tab1:
     
     status_values=['Won','Draft','To be approved','Lost','Not lost for AM','Wonderful','Revised','Offered','Offerable']
     item_type_values=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
     country_values=[28.0,25.0,30.0,32.0,38.0,78.0,27.0,77.0,113.0,79.0,26.0,39.0,40.0,84.0,80.0,107.0,89.0,44.8932136113145]
     application_values=[10.0,41.0,28.0,59.0,15.0,4.0,38.0,56.0,42.0,26.0,27.0,19.0,20.0,66.0,29.0,22.0,40.0,25.0,67.0,79.0,3.0,99.0,2.0,5.0,39.0,69.0,70.0,65.0,58.0,68.0]
     product_ref_values=[1670798778,1668701718,628377,640665,611993,1668701376,164141591,1671863738,1332077137,640405,1693867550,1665572374,1282007633,1668701698,628117,1690738206,628112,
                    640400,1671876026,164336407,164337175,1668701725,1665572032,611728,1721130331,1693867563,611733,1690738219,1722207579,929423819,1665584320,1665584662,1665584642]
with st.form("Data"):
     col1,col2=st.columns(2)
     with col1:
          st.write(' ')
          status=st.selectbox("Status:",status_values,key=1)
          item_type=st.selectbox("Itemtype",item_type_values,key=2)
          country=st.selectbox("Country",country_values,key=3)
          application=st.selectbox("Application",application_values,key=4)
          product_ref=st.selectbox("Product",product_ref_values,key=5)

     with col2:
          quantity_tons_log= st.text_input("Enter Quantity tons (Min:1.0,Max:20.72326583694641)")
          thickness_log= st.text_input("Thickness (Min:0.18,Max:5.991464547107982)")
          width= st.text_input("Width (Min:1.0,Max:2990.0)")
          customer=st.text_input("Customer (Min:12458.0,Max:2147483647.0)")
          submit_button=st.form_submit_button(label="PREDICT SELLING PRICE")

          if submit_button:

               import pickle
               with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\model.pkl",'rb') as file:
                    loaded_model=pickle.load(file)

               with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\scaler.pkl",'rb') as file1:
                    scalar_loaded=pickle.load(file1)
     
               with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\type.pkl",'rb') as file1:
                    type_loaded=pickle.load(file1)
               
               with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\status.pkl",'rb') as file1:
                    status_loaded=pickle.load(file1)

               user_input=np.array([[quantity_tons_log,status,item_type,country,application,thickness_log,width,customer,product_ref]])
               user_input_ohe=status_loaded.transform(user_input[:,[1]]).toarray()
               user_input_ohe2=type_loaded.transform(user_input[:,[2]]).toarray()
               user_input=np.concatenate((user_input[:,[0,3,4,5,6,7,8]],user_input_ohe,user_input_ohe2),axis=1)
               user_input1=scalar_loaded.transform(user_input)
               user_prediction=loaded_model.predict(user_input1)
               st.write(":blue[Predicted selling price:]", np.exp(user_prediction))

with tab2:

     with st.form("Data2"):
          col1,col2=st.columns(2)

          with col1:
               squantity_tons_log= st.text_input("Enter Quantity tons (Min:1.0,Max:20.72326583694641)")
               sthickness_log= st.text_input("Thickness (Min:0.18,Max:5.991464547107982)")
               swidth= st.text_input("Width (Min:1.0,Max:2990.0)")
               scustomer=st.text_input("Customer (Min:12458.0,Max:2147483647.0)")
               selling_price=st.text_input("Sellingprice (Min:1.0,Max:100001015.0)")

          with col2:
               sitem_type=st.selectbox("Itemtype",item_type_values,key=21)
               scountry=st.selectbox("Country",country_values,key=31)
               sapplication=st.selectbox("Application",application_values,key=41)
               sproduct_ref=st.selectbox("Product",product_ref_values,key=51)
               ssubmit_button=st.form_submit_button(label="PREDICT STATUS")

               if ssubmit_button:

                    import pickle     
                    with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\dcrmodel.pkl",'rb') as file:
                         sloaded_model=pickle.load(file)

                    with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\dcrscaler.pkl",'rb') as file1:
                         sscalar_loaded=pickle.load(file1)
          
                    
                    
                    with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\dcrstatus.pkl",'rb') as file1:
                         sstatus_loaded=pickle.load(file1)

               
                    user_input=np.array([[squantity_tons_log,sitem_type,scountry,sapplication,sthickness_log,swidth,scustomer,sproduct_ref,selling_price]])
                    user_input_ohe=sstatus_loaded.transform(user_input[:,[1]]).toarray()
                    
                    user_input=np.concatenate((user_input[:,[0,2,3,4,5,6,7,8]],user_input_ohe),axis=1)
                    user_input1=sscalar_loaded.transform(user_input)

                    user_prediction=sloaded_model.predict(user_input1)
                    if user_prediction==1:

                         st.write(" :blue[Predicted Staus: Won]" )
                    
                    else:
                         st.write(" :blue[Predicted Staus: Lost]" )

with st.sidebar:
    st.title(":blue[INDUSTRIAL COPPER MODELING]")
    st.header(":red[STEPS FOLLOWED]")
    st.caption("Explored skewness and outliers in the dataset")
    st.caption("Transformed the data into a suitable format and perform any necessary cleaning and pre-processing steps")
    st.caption("Created ML Regression model which predicts continuous variable ‘Selling_Price’")
    st.caption("Developed ML Classification model which predicts Status: WON or LOST ")
    st.caption("Created a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost)")
    st.caption("Read,update and delete options created in the streamlit for data modification")
    st.header(":red[TECHNOLOGIES USED]")
    st.caption("Python scripting,Pandas,Numpy,Seaborn,Matplotlib,Data Preprocessing,EDA, Streamlit")






               
          


     



