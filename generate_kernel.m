function  [train_within, test_train] = generate_kernel(train_data, test_data, type, scale)

% data is #edge by #subjects

K = size(train_data,1);
mu1 = mean(train_data, 2);
std1 = std(train_data, [],2);

no_train = size( train_data, 2);
no_test = size(test_data,2);

% normalize the data;
norm_train = zeros(K, no_train);
norm_test = zeros(K, no_test);

for i = 1: K
    if( std1(i)~=0)
        norm_train(i,:) = (train_data(i,:)-mu1(i))/std1(i);
        norm_test(i,:) = (test_data(i,:)-mu1(i))/std1(i);
    end
end
    
if(strcmp(type, 'corr'))
   train_within = corr(norm_train, norm_train);
   test_train = corr( norm_test, norm_train);
elseif(strcmp(type,'Exponential'))
    dist_train= L2_distance(norm_train, norm_train);
    train_within = exp(-1*scale*dist_train/K);
    dist_test = L2_distance(norm_test, norm_train);
    test_train = exp(-1*scale*dist_test/K);
elseif(strcmp(type,'Gaussian'))
    dist_train= L2_distance(norm_train, norm_train);
    train_within = exp(-1*scale*(dist_train.^2)/K);
    dist_test = L2_distance(norm_test, norm_train);
    test_train = exp(-1*scale*(dist_test.^2)/K);
else
    error('Unknown kernel type: %s', type);
end