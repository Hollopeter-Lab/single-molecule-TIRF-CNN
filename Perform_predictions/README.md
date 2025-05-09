Scripts are numbered in the order that they should be performed.

1_farred_trace_extract_normalize_convert.py prompts the user to upload .MAT files
produced by the analyze_batch.m function from https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m.
All farred spots that are colocalized with green spots will have their spot location 
and corresponding traces extracted. The traces will then be normalized by Z-score. The 
data will then be saved in a format compatible with downstream analysis via Python.

2_farred_predict.py imports a user selected Pytorch file containing the trained CNN. The
user then selects the .MAT file from the previous step.
