These scripts assume you have followed the protocol described in the README.md of the 
Perform_predictions directory.

To visualize data, you first need to create a .csv file organized such that each row
represents data from a colocalized green and far-red spot. Columns will contain the
farRed_source_file, farRed_spotLocation, predicted_steps, farred_trace, 
green_source_file, green_spotLocation, and green_trace. This will allow the data
visualizer to display traces from colocalized spots side-by-side. To create this .csv
file, use data_prep_for_visualization.py. This script prompts user to first select 
the excel file containing far-red predictions matched to their colocalized green
traces. The file should 
