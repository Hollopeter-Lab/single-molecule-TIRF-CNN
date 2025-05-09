GREEN_train_many_cnn.py script is meant to test nine 
combinations of different learning rates and batch sizes 
in each of the 39 model architectures defined in the 
Model_architectures repository.

GREEN_compile_classification_reports.py is meant to combine
metrics from all test models into one excel sheet to facilitate
choosing the best performing configurations of model architecture
with training protocol.

GREEN_K_fold_cross_validation_architecture_assessment.py is
meant to perform 5-fold cross validation on a single
configuration and collect performance metrics for each fold. Use
this to evaluate a final selection of configurations and choose 
a final architecture and training protocol configuration.

GREEN_train_final_model.py is meant to train a single architecture 
and training protocol configuration on 100% of a training dataset. It
does not provide meaningful metrics on the final trained model. Rather,
metrics from the K-fold cross validation associated with the final
architecture and training protocol configuration should be reported 
for the model performance. 
