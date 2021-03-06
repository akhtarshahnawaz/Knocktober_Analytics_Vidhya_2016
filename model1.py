import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import operator
import xgboost as xgb
import os

model_name = os.path.basename(__file__).split(".")[0]
np.random.seed(0)
###############################
print "Setting up Classifier"
###############################
def Classifier(x_train,x_test,y_train,y_test, test):
	dtrain = xgb.DMatrix(x_train, y_train)
	dvalid = xgb.DMatrix(x_test, y_test)
	params = {
		"objective": "binary:logistic",
		"booster": "gbtree",
		"eta": 0.3,
		"max_depth": 6,
		"eval_metric":"auc",
		"min_child_weight": 1.5,
		"gamma":0.0,
		"colsample_bytree": 0.7,
		"colsample_bylevel":0.7,
		"subsample":0.7,
		"tree_method":"exact",
		"alpha": 0.3,
		"lambda":1.3,
		"silent": 1,
		"num_parallel_tree":3,
		"seed":6656,
		"missing": -9999
	}
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	gbm = xgb.train(params, dtrain, 1200, evals=watchlist, verbose_eval=True, early_stopping_rounds =30)
	auc_score = roc_auc_score(y_test,gbm.predict(xgb.DMatrix(x_test), ntree_limit = gbm.best_iteration))
	return auc_score, gbm.best_iteration, params, gbm.predict(xgb.DMatrix(test), ntree_limit = gbm.best_iteration)

###############################
print "Loading Data"
###############################
input_dir = "../data/"
test = pd.read_csv(input_dir+"test.csv", parse_dates = ["Registration_Date","CD_Camp_Start_Date","CD_Camp_End_Date","PD_First_Interaction"])
train = pd.read_csv(input_dir+"train.csv", parse_dates = ["Registration_Date","CD_Camp_Start_Date","CD_Camp_End_Date","PD_First_Interaction"])

patient_id = test.Patient_ID
health_camp_id = test.Health_Camp_ID
target = train.target

print train.shape, test.shape

##################################
print "Identifying Columns"
##################################
columns = [c for c in test.columns.values.tolist() if c not in ["Patient_ID","Health_Camp_ID"]]

categorical_columns = [c for c in columns if train[c].dtype=="O"]
date_columns = [c for c in columns if train[c].dtype=="datetime64[ns]"]
general_columns = [c for c in columns if train[c].dtype not in ["datetime64[ns]","O"]]

##################################
print "Generating Features"
##################################
data = pd.concat([train,test]).reset_index(drop=True)

# Generating Date Features
#-------------------------
for col in date_columns:
	data[col+"_month"] = data[col].dt.month
	data[col+"_year"] = data[col].dt.year
	data[col+"_day"] = data[col].dt.day
	data[col+"_weekday"] = data[col].dt.weekday

gen_date_columns = [[c+"_month",c+"_year",c+"_day",c+"_weekday"] for c in date_columns]
gen_date_columns = sum(gen_date_columns,[])

# Preventing Overfitting by removing variables that aren't common between train/test
#-----------------------------------------------------------------------------------
data["CD_Category2"][data.CD_Category2 == "B"] =np.nan
data["CD_Category3"][data.CD_Category3 == 1] =np.nan

data["Var1"].loc[data["Var1"].isin(data["Var1"].value_counts()[data["Var1"].value_counts()==1].index)] = np.nan
data["Var2"].loc[data["Var2"].isin(data["Var2"].value_counts()[data["Var2"].value_counts()==1].index)] = np.nan
data["Var4"].loc[data["Var4"].isin(data["Var4"].value_counts()[data["Var4"].value_counts()==1].index)] = np.nan
data["Var5"].loc[data["Var5"].isin(data["Var5"].value_counts()[data["Var5"].value_counts()==1].index)] = np.nan


# Generating time features about camp and Person
#-----------------------------------------------
data["time_since_interaction"] = (data["Registration_Date"]-data["PD_First_Interaction"])/np.timedelta64(1, 'D')
data["time_since_camp_start"] = (data["Registration_Date"]-data["CD_Camp_Start_Date"])/np.timedelta64(1, 'D')
data["time_before_camp_ends"] = (data["CD_Camp_End_Date"] - data["Registration_Date"])/np.timedelta64(1, 'D')

# Generating features about number of people in camp
#----------------------------------------------------
data["num_people_in_camp"] = data["Health_Camp_ID"].map(data.groupby("Health_Camp_ID")["Patient_ID"].nunique())
data["num_people_in_camp_today"] = data["Registration_Date"].map(data.groupby("Registration_Date")["Patient_ID"].nunique())


# Generating features about last visits
#----------------------------------------------------
data["time_since_last_visit"] = data.sort_values(by = "Registration_Date").groupby("Patient_ID")["Registration_Date"].diff().fillna(0)/np.timedelta64(1, 'D')
data["time_since_first_visit"] = (data["Registration_Date"] - data["Patient_ID"].map(data.groupby("Patient_ID")["Registration_Date"].apply(lambda x: np.min(x))))/np.timedelta64(1, 'D')
data["time_since_last_visit_in_this_camp_today"] = data.sort_values(by =["Patient_ID","Health_Camp_ID","Registration_Date"])["Registration_Date"].diff().fillna(0)/np.timedelta64(1, 'D')

# Generating features about number of visits in camps
#----------------------------------------------------
data["number_of_visits_in_this_camp"] = data.sort_values(by = "Registration_Date").groupby(["Health_Camp_ID","Patient_ID"])["Registration_Date"].cumcount()
data["number_of_visits_in_this_camp_type1"] = data.sort_values(by = "Registration_Date").groupby(["Patient_ID","CD_Category1"])["Registration_Date"].cumcount()
data["number_of_visits_in_this_camp_type2"] = data.sort_values(by = "Registration_Date").groupby(["Patient_ID","CD_Category2"])["Registration_Date"].cumcount()
data["number_of_visits_in_this_camp_type3"] = data.sort_values(by = "Registration_Date").groupby(["Patient_ID","CD_Category3"])["Registration_Date"].cumcount()


# Generating features about last Score received by Person
#--------------------------------------------------------
data["last1_score_camp1"] = data.loc[data.CD_Category1 == "First"].groupby(["Patient_ID"])["F_Health_Score"].shift(1)
data["last1_score_camp2"] = data.loc[data.CD_Category1 == "Second"].groupby(["Patient_ID"])["S_Health_Score"].shift(1)
data["last2_score_camp1"] = data.loc[data.CD_Category1 == "First"].groupby(["Patient_ID"])["F_Health_Score"].shift(2)
data["last2_score_camp2"] = data.loc[data.CD_Category1 == "Second"].groupby(["Patient_ID"])["S_Health_Score"].shift(2)

# Generating features about number of visits by a person
#--------------------------------------------------------
data["number_of_visits_today"] = data.sort_values(by = "Registration_Date").groupby(["Registration_Date","Patient_ID"])["Registration_Date"].cumcount()
data["visits_till_now"] = data.sort_values(by = "Registration_Date").groupby("Patient_ID")["Registration_Date"].cumcount()

# Generating features about when next camp is
#--------------------------------------------------------
camp_starts_on = data.groupby("Health_Camp_ID")["CD_Camp_Start_Date"].first().sort_values()
camp_ends_on = data.groupby("Health_Camp_ID")["CD_Camp_End_Date"].last().sort_values()
gaps_between_camp = camp_starts_on.shift(-1) - camp_ends_on
data["next_camp_in"] = data["time_before_camp_ends"] + (data["Health_Camp_ID"].map(gaps_between_camp))/np.timedelta64(1, 'D')


extra_features = []
extra_features += ["time_since_interaction","time_since_camp_start","time_before_camp_ends"]
extra_features += ["num_people_in_camp","num_people_in_camp_today"]
extra_features += ["time_since_last_visit","time_since_first_visit","time_since_last_visit_in_this_camp_today"]
extra_features += ["number_of_visits_in_this_camp","number_of_visits_in_this_camp_type1","number_of_visits_in_this_camp_type2","number_of_visits_in_this_camp_type3"]
extra_features += ["last1_score_camp1","last1_score_camp2","last2_score_camp1","last2_score_camp2"]
extra_features += ["number_of_visits_today"]
extra_features += ["visits_till_now"]
extra_features += ["next_camp_in"]

##################################
print "Encoding Categoricals"
##################################
for col in categorical_columns:
	data[col] = LabelEncoder().fit_transform(data[col].fillna("Is Null"))

ohe = OneHotEncoder(sparse = False)
ohe_data = ohe.fit_transform(data[categorical_columns])
ohe_data = pd.DataFrame(ohe_data)

train_cat = ohe_data[:train.shape[0]].reset_index(drop=True)
test_cat = ohe_data[train.shape[0]:].reset_index(drop=True)

train = data[:train.shape[0]].reset_index(drop=True)
test = data[train.shape[0]:].reset_index(drop=True)

train = pd.concat([train[general_columns+gen_date_columns+extra_features], train[["Registration_Date","target"]], train_cat], axis=1).reset_index(drop=True)
test = pd.concat([test[general_columns+gen_date_columns+extra_features], test_cat], axis=1).reset_index(drop=True)
train = train.fillna(-9999)
test = test.fillna(-9999)

del data, train_cat, test_cat
##################################
print "Generating Validation Set"
##################################
train = train.sort_values(by = "Registration_Date").reset_index(drop=True)

x_train , x_test = train[0:-15000], train[-15000:]
y_train , y_test = train[0:-15000].target, train[-15000:].target
x_train , x_test = x_train.drop(["Registration_Date","target"], axis=1), x_test.drop(["Registration_Date","target"], axis=1)

target = train.target
train = train.drop(["Registration_Date","target"], axis=1)

print train.shape, test.shape

##################################
print "Cross Validating"
##################################
auc_score, n, params, prediction = Classifier(x_train,x_test,y_train,y_test, test)

##################################
print "Training"
##################################
dtrain = xgb.DMatrix(train, target)
gbm = xgb.train(params, dtrain, n)
prediction= gbm.predict(xgb.DMatrix(test), ntree_limit = gbm.best_iteration)
auc_score = str(auc_score).split(".")[1]
##################################
print "Saving"
##################################
pd.DataFrame({"Patient_ID":patient_id,"Health_Camp_ID":health_camp_id, "Outcome": prediction}).to_csv("csv/"+auc_score+"_"+model_name+".csv", index = False)


