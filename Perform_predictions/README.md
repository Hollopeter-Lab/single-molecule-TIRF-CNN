Scripts are numbered in the order that they should be performed.

1_farred_trace_extract_normalize_convert.py prompts the user to upload .MAT files
produced by the analyze_batch.m function from https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m.
All farred spots that are colocalized with green spots will have their spot location 
and corresponding traces extracted. The traces will then be normalized by Z-score. The 
data will then be saved in a format compatible with downstream analysis via Python.

2_farred_predict.py requires a user to change the code. The user should select the model 
architecture (line 10) according to their training results. The scripts then imports a 
user selected Pytorch file containing the trained CNN weights. The user then selects the 
.MAT file containing the normalized far-red traces. 
