
import streamlit
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


 
streamlit.set_page_config(page_title="Drought Prediction System", layout="wide")


with streamlit.sidebar:
    streamlit.title("Drought Prediction")
    streamlit.header("Project Overview")
    streamlit.write("""
    Early warning drought prediction system using
    climate features such as temperature, humidity,
    precipitation, solar radiation and SPEI index.

    
    """)
    streamlit.subheader("🎯 Goal : Support farmers & decision makers in planning")
    streamlit.divider()
    streamlit.write("📌 *Enter input values and  click Show result.*")


streamlit.title("Drought Prediction System")


data=pd.read_csv("C:/Users/Sreeraj/OneDrive/Desktop/ML_project/Drought_Prediction/stage_4_drought_dataset.csv")
df=pd.DataFrame(data)
df=df.drop('row_id',axis=1)
X=df.drop('label',axis=1)
y=df['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
nu=GaussianNB()
model=nu.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))





def input_features():
    RH2M= streamlit.number_input("Relative humidity at 2 meters(RH2M) (%) ",
                                 df['RH2M'].min(),df['RH2M'].max(),df['RH2M'].mean())       
                                                   
    T2M_MAX= streamlit.number_input("Maximum temperature at 2 meters(T2M_MAX) (°C)"
                                    ,df['T2M_MAX'].min(),df['T2M_MAX'].max(),df['T2M_MAX'].mean())  
                           
    T2M_MIN= streamlit.number_input("Minimum temperature at 2 meters(T2M_MIN) (°C)",
                                    df['T2M_MIN'].min(),df['T2M_MIN'].max(),df['T2M_MIN'].mean()) 
                       
    WS2M = streamlit.number_input("Wind speed at 2 meters(WS2M) (m/s)",
                                  df['WS2M'].min(),df['WS2M'].max(),df['WS2M'].mean())  
                     
    T2M= streamlit.number_input("Mean temperature at 2 meters above surface(T2M) (°C)",
                                df['T2M'].min(),df['T2M'].max(),df['T2M'].mean())     
                       
    ALLSKY_SFC_SW_DWN=  streamlit.number_input("All Sky Surface Shortwave Downward Irradiance(ALLSKY_SFC_SW_DWN) (W/m² or kW-hr/m²/day)",
                                               df['ALLSKY_SFC_SW_DWN'].min(),df['ALLSKY_SFC_SW_DWN'].max(),
                                               df['ALLSKY_SFC_SW_DWN'].mean())     
                          
    PRECTOTCORR= streamlit.number_input("Corrected Precipitation(PRECTOTCORR) (mm or mm/day)",
                                        df['PRECTOTCORR'].min(),df['PRECTOTCORR'].max(),df['PRECTOTCORR'].mean())
             
    spei=streamlit.number_input("Standardized Precipitation Evapotranspiration Index(SPEI)",
                                df['spei'].min(),df['spei'].max(),df['spei'].mean()) 
    
    lat_sin=streamlit.number_input("Latitude Sin(lat_sin)",df['lat_sin'].min(),df['lat_sin'].max(),df['lat_sin'].mean())

    lat_cos=streamlit.number_input("Latitude cos(lat_cos)",df['lat_cos'].min(),df['lat_cos'].max(),df['lat_cos'].mean())

    lon_sin=streamlit.number_input("Longitude sin(lon_sin)",df['lon_sin'].min(),df['lon_sin'].max(),df['lon_sin'].mean())

    lon_cos=streamlit.number_input("Longitude cos(lon_cos)",df['lon_cos'].min(),df['lon_cos'].max(),df['lon_cos'].mean())

    month_sin=streamlit.number_input("Month sin(month_sin)",df['month_sin'].min(),df['month_sin'].max(),df['month_sin'].mean())
    
    month_cos=streamlit.number_input("Month cos(month_cos)",df['month_cos'].min(),df['month_cos'].max(),df['month_cos'].mean())
    
        


    data={
        "RH2M":RH2M,
        "T2M_MAX":T2M_MAX,
        "T2M_MIN":T2M_MIN,
        "WS2M":WS2M,
        "T2M":T2M,
        "ALLSKY_SFC_SW_DWN":ALLSKY_SFC_SW_DWN,
        "PRECTOTCORR":PRECTOTCORR,
        "spei":spei,
        "lat_sin":lat_sin,
        "lat_cos":lat_cos,
        "lon_sin":lon_sin,
        "lon_cos":lon_cos,
        
        "month_sin":month_sin,
        "month_cos":month_cos,
    }

    features=pd.DataFrame(data,index=[0])

    return features

input_df=input_features()

input_scaled=scaler.transform(input_df)

result=model.predict(input_scaled)

if streamlit.button("Show result"):
    final_result=" 🌿 NO Drought" if result[0]==0 else " ⚠️ Drought"    

    streamlit.success(final_result)