Drought Prediction - Supervised ML Classification
--------------------------------------------------
A Supervised machine learning project that predict drought conditions using climate,hydrological,spatial, and seasonal data.

------

Overview
--------
A Machine learning based drought prediction system build using global climate data.
The project combines data analysis,model training and a streamlit interface to provide real time drought predictions.

Objective
--------- 
To classify drought conditions using Supervised Machine learning models based on key climate variables.
To build a binary classification model that can accurately predict drought events based on historical climate data.

Dataset
--------
Source : Kaggle -Drought prediction dataset

Link : https://www.kaggle.com/datasets/vijayaragulvr/drought-prediction-dataset

Time period: Jan 2015-Dec 2023(108 months)

Target :
   0 --> No Drought
   1 --> Drought

Tools & Libraries
----------------
python

pandas

scikit-learn

streamlit(intercative web app)

matplotlib,seaborn(visualization)


Input Features
-------------
Climate and Atmospheric Features
--------------------------------
RH2M :
Relative Humidity measured at 2 meters above ground level(shows how much moisture is present in the air at 2 m above the ground)

   lower values indicates increased evaporation.
        
T2M_MAX,T2M_MIN,T2M :

Maximum,Minimum,Average air temperature at 2 meters above the surface.

   High tempeature increase water loss from soil.
   
WS2M(Wind speed at 2 meters) :

Wind speed near the ground that affects how fast moisture evaporates.

ALLSKY-SFC-SW-DWN(All-sky surface shortwave downward irradiance) :

Amount of sunlight reaching the ground,which increases heating and evaporation.


Hydrological Features
---------------------
Prectotcorr(corrected precipitation) :

Total rainfall after correction(less rainfall overtime leads to drought).

SPEI(Standardized Precipitation Evapotranspiration index):

A drought index based on rainfall and  temperature .

   Negative values shows dry conditions.


Spatial Features
----------------

lat_sin,lat_cos : Encoded latitude values that help the model understand location.


lon_sin,lon_cos : Encoded longitude values that help identify regional patterns.

Seasonal Features
----------------

month_sin,month_cos : Encoded month values used to represent seasons like monsoons and dryperiod.


Predictive Framework
--------------------
Type : Supervised classification.

Algorithm : Naive Bayes.

Prediction target : Drought occurence (0= No Drought, 1 = Drought)

Preprocessing: feature scaling using standard Scaler.


Evaluation
--------------

The model performance is evaluatd using:
----------------------------------------

Accuracy score

Confusion matrix

Precision

recall

f1 score


interactive interface(streamlit web app)
-----------------------

Users can input climate,hydrological,spatial and seasonal features.

Provides real-time drought prediction.

Clear,numeric inputs for all entries.

   How to run
   ---------
   Clone the repository
   
   Install required packages
   
   Run the Streamlit app : streamlit run app.py



