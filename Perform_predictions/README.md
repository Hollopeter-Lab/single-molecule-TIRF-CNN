Scripts are numbered in the order that they should be performed.

1_farred_trace_extract_normalize_convert.py prompts the user to upload .mat files
produced by the analyze_batch.m function from https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m.
All far-red spots that are colocalized with green spots will have their spot location 
and corresponding traces extracted. The traces will then be normalized by Z-score. The 
data will then be saved in a format compatible with downstream analysis via Python.

2_farred_predict.py requires a user to input the exact model architecture into the code 
(line 10) according to their training results. The script then prompts the user to select
a Pytorch file containing the trained CNN weights. The script prompts the user to select the 
.mat file containing the normalized far-red traces. Predictions in the assigned number
of classes will be made and saved. Multiple files will be saved: "..._predictions.csv" 
contains the predicted step for each trace and associated probabilities for each class, 
"..._predicted_steps_gt0.csv" only contains predicted steps and associated spot location 
data for traces where the predicted class > 0, "..._final_model_summary.csv" contains 
class-wise summations of all traces for each sample, "..._final_model_summaries.xlsx" 
contains class-wise summations of traces for each sample filtered by probability thresholds.
To simplify downstream analysis, the file "..._predicted_steps_gt0.csv" will be used.

3_green_trace_extract_normalize_convert.py prompts the user to upload .mat files
produced by the analyze_batch.m function from https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m.
All green spots that are colocalized with far-red spots will have their spot location 
and corresponding traces extracted. The traces will then be normalized by Z-score. The 
data will then be saved in a format compatible with downstream analysis via Python.

4_compile_green_traces_with_associated_farred_predictions.py prompts the user to upload
the .csv file "..._predicted_steps_gt0.csv" from the 2_farred_predict.py output. The user
is then prompted to upload the .mat file containing their extracted and normalized green 
traces. The script then matches and appends the green traces to the appropriate far-red 
predictions based on shared spot locations. A file called 
"...farred matched to green traces.csv" is saved.

5_green_predict_and_summarize.py requires a user to input the exact model architecture into the code 
(line 10) according to their training results. The script then prompts the user to select
a Pytorch file containing the trained CNN weights. The script prompts the user to select the 
.csv file containing the far-red predictions matched with the green traces. Predictions in the
assigned number of classes will be made and saved. Two files called "...green predictions matched with farred predictions.xlsx"
and "...green predictions matched with farred predictions.csv" will be saved. The .xlsx will contain
a summary sheet showing the number of green traces predicted in each class, a detailed_predictions 
sheet showing each far-red spot and its prediction paired with each green spot and its prediction, a 
green_step_counts_per_file sheet showing the green predictions broken down per sample, and a
farRed_dist_if_green_1 showing the distribution of far-red predictions per sample when the green
trace was classified as having one-step.

Visit the Visualize_predictions folder to see scripts on how to view far-red traces
with their predictions alongside corresponding colocalized green traces.
