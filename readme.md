############################
# Model Runs in three steps
############################

1. Run generator.py to generate "train.csv" and "test.csv" files. Use appropriate input and output directories.

2. Run files "model{1:N}.py" to make predictions. These predictions will be saved in csv folder and will be automatically taken by ensembler.py

3. Run "ensembler.py" for final ensemble.