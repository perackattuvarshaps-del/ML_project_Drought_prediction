
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
    RH2M= streamlit.number_input("Relative humidity at 2 meters(RH2M) (%)")
                            
                                                   
    T2M_MAX= streamlit.number_input("Maximum temperature at 2 meters(T2M_MAX) (°C)")
                                      
                           
    T2M_MIN= streamlit.number_input("Minimum temperature at 2 meters(T2M_MIN) (°C)")
                                
                       
    WS2M = streamlit.number_input("Wind speed at 2 meters(WS2M) (m/s)")
                                  
                     
    T2M= streamlit.number_input("Mean temperature at 2 meters above surface(T2M) (°C)")
                                     
                       
    ALLSKY_SFC_SW_DWN=  streamlit.number_input("All Sky Surface Shortwave Downward Irradiance(ALLSKY_SFC_SW_DWN) (W/m² or kW-hr/m²/day)")
                                                 
                          
    PRECTOTCORR= streamlit.number_input("Corrected Precipitation(PRECTOTCORR) (mm or mm/day)")
                                        
             
    spei=streamlit.number_input("Standardized Precipitation Evapotranspiration Index(SPEI)")
                                
    
    lat_sin=streamlit.number_input("Latitude Sin(lat_sin)")
    

    lat_cos=streamlit.number_input("Latitude cos(lat_cos)")
                                   

    lon_sin=streamlit.number_input("Longitude sin(lon_sin)")
                                   
                        

    lon_cos=streamlit.number_input("Longitude cos(lon_cos)")
                                   

    month_sin=streamlit.number_input("Month sin(month_sin)")
                                     
    
    month_cos=streamlit.number_input("Month cos(month_cos)")
                                    
    
        


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
    final_result="  NO Drought" if result[0]==0 else " Drought"    

    streamlit.success(final_result)