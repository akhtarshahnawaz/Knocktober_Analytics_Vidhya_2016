import numpy as np
import pandas as pd
import os
import random

#####################
print "Loading Data"
#####################
input_dir = os.path.realpath("csv/")
files = os.listdir(input_dir)
prediction_files = [pd.read_csv(os.path.join(input_dir,f)) for f in files if "model" in f]

#########################
print "Ensembling"
#########################
predictions = [f.Outcome for f in prediction_files]
predictions = pd.DataFrame(predictions).mean(axis=0).reset_index(drop=True)

#########################
print "Saving Files"
#########################
output = prediction_files[0]
output["Outcome"] = predictions
output.to_csv("csv/ensemble.csv",index=False)