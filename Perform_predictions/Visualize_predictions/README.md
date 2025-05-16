These scripts assume you have followed the protocol described in the README.md of the 
Perform_predictions directory.

To visualize data, you first need to create a .csv file organized such that each row
represents data from a colocalized green and far-red spot. Columns will contain the
farRed_source_file, farRed_spotLocation, predicted_steps, farred_trace, 
green_source_file, green_spotLocation, and green_trace. This will allow the data
visualizer to display traces from colocalized spots side-by-side. To create this .csv
file, use data_prep_for_visualization.py. This script prompts user to first select 
the excel file containing far-red predictions matched to their colocalized green
traces. This file is the output of 4_compile_green_traces_with_associated_farred_predictions.py.
The script then prompts the user to select the .mat file containing all far-red traces.
This file is the output of 1_farred_trace_extract_normalize_convert.py. Far-red traces will
then be extracted from the .mat file according to spot locations cataloged in the excel file to
construct the final .csv containing all data described above.

data_visualizer.py will open a gui where the user may select the .csv file of interest.
The green and far-red traces for one colocalization event will be displayed along with the
index number for the data within the .csv file, pixel coordinates of each spot, the source file, 
and predicted steps for the far-red trace. The user may filter data by the number of predicted 
steps in the far-red trace.

Additional features may be added in the future.

NOTE: The data this tool will visualize consists of all colocalization events where the
far-red trace was not rejected. These data will not have been analyzed for their green
traces and filtered to also remove colocalization events based upon green trace rejection.
The user may want to adjust the script to allow further filter the data they are visualizing. 
