function [pred, estimate] = kernel_prediction(train_data, test_data, train_score, lambda,type, scale)
% pred is the prediction of based on the test data
% estimate is the estimate from the training data

[train_within, test_train] = generate_kernel(train_data, test_data, type, scale);

no_train = size(train_data,2);

pred = test_train*(( train_within +lambda*eye(no_train))\train_score);

estimate = train_within*(( train_within +lambda*eye(no_train))\train_score);