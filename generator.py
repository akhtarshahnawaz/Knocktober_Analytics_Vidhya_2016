import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb

np.random.seed(0)

###############################
print "Loading Data"
###############################
input_dir = "../data/"
test = pd.read_csv(input_dir+"Test_D7W1juQ.csv", parse_dates = ["Registration_Date"])
train = pd.read_csv(input_dir+"Train/Train.csv", parse_dates = ["Registration_Date"])

health_camp_detail = pd.read_csv(input_dir+"Train/Health_Camp_Detail.csv", parse_dates = ["Camp_Start_Date","Camp_End_Date"])
patient_detail = pd.read_csv(input_dir+"Train/Patient_Profile.csv", parse_dates = ["First_Interaction"])

first_health_camp = pd.read_csv(input_dir+"Train/First_Health_Camp_Attended.csv")
second_health_camp = pd.read_csv(input_dir+"Train/Second_Health_Camp_Attended.csv")
third_health_camp = pd.read_csv(input_dir+"Train/Third_Health_Camp_Attended.csv")

patient_id = test.Patient_ID
health_camp_id = test.Health_Camp_ID

print train.shape, test.shape
###############################
print "Preprocessing Tables"
###############################
second_health_camp.rename(columns={"Health Score":"Health_Score"}, inplace=True)

first_health_camp.columns = [s if s in ["Patient_ID","Health_Camp_ID"] else "F_"+s for s in first_health_camp.columns]
second_health_camp.columns = [s if s in ["Patient_ID","Health_Camp_ID"] else "S_"+s for s in second_health_camp.columns]
third_health_camp.columns = [s if s in ["Patient_ID","Health_Camp_ID"] else "T_"+s for s in third_health_camp.columns]

health_camp_detail.columns = [s if s in ["Health_Camp_ID"] else "CD_"+s for s in health_camp_detail.columns ]
patient_detail.columns = [s if s in ["Patient_ID"] else "PD_"+s for s in patient_detail.columns]

###############################
print "Joining Tables"
###############################
# Joining "Patient Profile" & "Health Camp Details"
train = pd.merge(train, health_camp_detail, how='left',on="Health_Camp_ID", left_index=True)
train = pd.merge(train, patient_detail, how='left',on="Patient_ID", left_index=True)

test = pd.merge(test, health_camp_detail, how='left',on="Health_Camp_ID", left_index=True)
test = pd.merge(test, patient_detail, how='left',on="Patient_ID", left_index=True)

# Joining "Format1","Format2" & "Format3" to train
train = pd.merge(train, first_health_camp, how='left',on=["Patient_ID","Health_Camp_ID"], left_index=True)
train = pd.merge(train, second_health_camp, how='left',on=["Patient_ID","Health_Camp_ID"], left_index=True)
train = pd.merge(train, third_health_camp, how='left',on=["Patient_ID","Health_Camp_ID"], left_index=True)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print train.shape, test.shape
###############################
print "Generating Target"
###############################
train["target"] = np.zeros(train.shape[0])
train.loc[train.F_Health_Score.notnull(),"target"] =1
train.loc[train.S_Health_Score.notnull(),"target"] =1
train.loc[train.T_Number_of_stall_visited >=1,"target"] =1

########################################
print "Saving", train.shape, test.shape
########################################
train = train.drop(['F_Unnamed: 4'], axis=1)

train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)


