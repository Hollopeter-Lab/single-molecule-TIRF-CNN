Each script will perform the same function for either far-red or
green traces. .MAT files produced by the analyze_batch.m function from
https://github.com/dickinson-lab/SiMPull-Analysis-Software/blob/master/Static_Analysis/Data_Processing/analyze_batch.m
will be loaded. Far-red or green traces will be extracted from
the .MAT file, normalized by Z-score, and displayed in a small GUI where 
a user can assign photobleaching step values. These assigned labels will be saved 
and cataloged alongside the corresponding intensity trace. If the user skips a trace, 
it will be labeled as "0" which stands for rejected. If a trace has no steps, it can
also be labeled as "0" so that the trained CNN learns to reject messy
traces or traces with no photobleaching events. The assignments can be
saved as an Excel file. If the user plans on using other scripts from this
repository, they should export their labeled traces using the export button.
This will ensure compatibility with subsequent Python scripts.
