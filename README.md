# Linear SVM FeatureCloud App

## Description
A linear Support Vector Machine FeautureCloud app for binary classification.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local testing data (columns: features; rows: samples)

## Output
- train.csv containing the local training data
- test.csv containing the local testing data
- train_pred.csv containing the predicted labels for the local training data
- test_pred.csv containing the predicted labels for the local testing data
- train_y.csv containing the true labels for the local training data
- test_y.csv containing the true labels for the local testing data

## Workflows
Can be combined with the following apps:
- Various preprocessing apps (e.g. Normalization, Cross Validation, ...) 
- Various evaluation apps (e.g. Classification Evaluation, ...)

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_svm:
  input:
    train: "train.csv"
    test: "test.csv"
  output:
    train: "train_pred.csv"
    test: "test_pred.csv"
  format:
    sep: ","
    label: "y"
  split:
    mode: directory   # directory if cross validation was used before, else file
    dir: data   # data if cross validation app was used before, else .
  parameter:
    learning_rate: 0.00001   # learning rate parameter of the model
    regularization: 1   # regularization parameter of the model
    batch_size: -1   # batch size of the gradient descent update. -1 uses the enitre data for update
    local_steps: 100    # local gradient descent steps before aggregation
    convergence_rounds: 5   # number of rounds needed for convergence if the model cost does not improve
    convergence_factor: 0.999   # model cost multiplier for the previous round. 
```