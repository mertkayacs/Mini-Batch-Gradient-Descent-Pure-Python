# Mini-Batch-Gradient-Descent

Executed as follows in the directory where all files (.py , data.csv , model.txt) is in.:
python minibatchgradientdescent.py data.csv model.txt


model.txt must be in the form of:
a_0*col_1^d + a_1*col_2^(d-1) + ... + a_p
where each a i are the parameters of the model and col i are the columns of the csv file.

I have used the mean squared error as the error function.

Data file should look like :
,F1,F2,F3,F4,target
1,0.8377157496342837,0.6384139212761822,0.981834414945094,0.5957754304255635,0.057391640412137845
2,0.4027562538227457,0.05485865543468105,0.8918170342139552,0.9483892393317622,0.20508721034647115
3,0.653701528465938,0.6793248141356534,0.7950565588100541,0.3163774972481559,0.483799822699012

Model does not have to use all features.
Some of a_0(a_i's) can be 0.

Training / Test data is splitted %80 / %20 .
They are shuffled but it is optional(Check the main method to enable/disable).

Mert KAYA 2020
