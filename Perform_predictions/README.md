Scripts are numbered in the order that they should be performed.

1_farred_trace_extract_normalize_convert.py prompts the user to upload .MAT files
produced by the analyze_batch.m function from https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m.
All far-red spots that are colocalized with green spots will have their spot location 
and corresponding traces extracted. The traces will then be normalized by Z-score. The 
data will then be saved in a format compatible with downstream analysis via Python.

2_farred_predict.py requires a user to input the exact model architecture into the code 
(line 10) according to their training results. The script then imports a 
user selected Pytorch file containing the trained CNN weights. The user then selects the 
.MAT file containing the normalized far-red traces. Predictions in the assigned number
of classes will be made and saved. Multiple files will be saved: "..._predictions.csv" 
contains the predicted step for each trace and associated probabilities for each class, 
"..._predicted_steps_gt0.csv" only contains predicted steps and associated spot location 
data for traces where the predicted class > 0, "..._final_model_summary.csv" contains 
class-wise summations of all traces for each sample, "..._final_model_summaries.xlsx" 
contains class-wise summations of traces for each sample filtered by probability thresholds.
To simplify downstream analysis, the file "..._predicted_steps_gt0.csv" will be used.

3_green_trace_extract_normalize_convert.py prompts the user to upload .MAT files
produced by the analyze_batch.m function from https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m.
All green spots that are colocalized with far-red spots will have their spot location 
and corresponding traces extracted. The traces will then be normalized by Z-score. The 
data will then be saved in a format compatible with downstream analysis via Python.
