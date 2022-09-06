# pip install lifelines
import lifelines
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Loading the the survival un-employment data
survival_mort = pd.read_csv(r"C:\Users\optim\Desktop\MORTALITY PROJECT\Mortality.csv")
survival_mort.head()
survival_mort.describe()
# apply normalization techniques on Column 1



survival_mort["Age"].describe()

 #Age is referring to time 
T = survival_mort.Age

 #Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()


 #Fitting KaplanMeierFitter model on Time and mort_norms for death 
kmf.fit(T, event_observed=survival_mort.mort_norm)

 #Time-line estimations plot 
kmf.plot()

 #ver Multiple groups 
# For each group, here group is Age
survival_mort.Asthma.value_counts()

 #Applying KaplanMeierFitter model on Time and mort_norms for the group "1"
kmf.fit(T[survival_mort.Asthma==1], survival_mort.mort_norm[survival_mort.Asthma==1], label='1')
ax = kmf.plot()

 #Applying KaplanMeierFitter model on Time and mort_norms for the group "0"
kmf.fit(T[survival_mort.Asthma==0], survival_mort.mort_norm[survival_mort.Asthma==0], label='0')
kmf.plot(ax=ax)


