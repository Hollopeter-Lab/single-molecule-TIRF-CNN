FARRED_train_one_cnn.py is meant to test nine combinations of different 
learning rates and batch sizes in one of the 39 model architectures defined 
in the Model_architectures repository. You will need to alter the script
to loop through all 39 models or test each one manually. This will train a
CNN to recognize four classes. It will also merge tracels labeled as 3-step
or higher into a single class.

FARRED_compile_classification_reports.py is meant to combine
metrics from all tested configurations into one excel sheet to facilitate
choosing the best performing configurations of model architecture
with training protocol.

FARRED_K_fold_cross_validation_architecture_assessment.py is
meant to perform 5-fold cross validation on a single
configuration and collect performance metrics for each fold. Use
this to evaluate a final selection of configurations and choose 
a final architecture and training protocol configuration.

FARRED_train_final_model.py is meant to train a single architecture 
and training protocol configuration on 100% of a training dataset. It
does not provide meaningful metrics on the final trained model. Rather,
metrics from the K-fold cross validation associated with the final
architecture and training protocol configuration should be reported 
for the model performance.

data_loading.py is a utility function to allow proper import of .MAT
training data.
