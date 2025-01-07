% This script apply kernel ridge regression to predict a composite score
% It is derived from the main script "Testing_Set227_KRR.m"

clear
clc

% % define your data path
% data_path = '';
% % define your path to save the results
% save_path = '';

% load the matrix after the regression
load([data_path, 'trans_data_set227_mat_cov_regressed'], 'res_mat');

% load the original scores
load([data_path, 'trans_data_all_scores_set227'],'all_score');
% 146 = 124 + 16 + 6; other sub-scores + cognitive_scores + clinical scores

% load the regressed scores (for evaluation purpose)
load([data_path, 'trans_data_set227_allscores_cov_regressed'], 'all_res');

% load the covariates
load([data_path, 'trans_data_combined_update_cov_diag_set227'], 'cov_combined');

nan_flg = 1;
if( nan_flg ==1)
    % cognitive scores (16)
    datamat_orig = all_score(:, 125:140);
    datamat_res = all_res(:, 125:140);
elseif( nan_flg==2)
    % all the rest of the sub scores (124)
    datamat_orig = all_score(:, 1:124);
    datamat_res = all_res(:, 1:124);
elseif( nan_flg ==3)
    % clinical scores (6)
    datamat_orig = all_score(:, 141:146);
    datamat_res = all_res(:, 141: 146);
end

no_beh = 1; % one combined score;

% matrices after regression
multi_run_all_mats = res_mat;
no_runs = size(multi_run_all_mats, 3);
no_sub = size(multi_run_all_mats, 4);
no_nodes = size(multi_run_all_mats,1);

aa = ones( no_nodes, no_nodes);
aa_upp = triu(aa,1);
upp_id = find( aa_upp>0);
upp_len = length(upp_id);


% yeo_flg = 1; % yeo network
% yeo_flg = 3; % use DMN and Ev language
% yeo_flg = 2; % use the whole brain
yeo_flg = 0; % use the constructs
% yeo_flg =4; % use all 6 constructs combined

if( yeo_flg==0)
    % load the construct nodes
    load([data_path, 'construct6_network268_node_list'], 'check_roi');
    % replce the declarative memory construct with nodes from the
    % hippocampus from the Shen268 atlas        
    manual_dec = 1; 
    % including bilateral hippocampus nodes
    if(manual_dec==1)
        hip_id = [93:97 230:234];
        check_roi(:, 3) = 0;
        check_roi(hip_id, 3) = 1;
    end
    no_net = 6;
elseif( yeo_flg ==1)
    % Yeo's 7 network definition
    yeomap = dlmread([data_path, 'Parc268toYeo7netlabel']);
    
    no_net = size( yeomap, 2);
    check_roi = yeomap;
elseif( yeo_flg ==2)
    % use the whole brain;
elseif( yeo_flg ==3)
    % use the dmn and Ev's language network
    file_name = 'Shen268_10network';
   
    n_label = dlmread([data_path, file_name], '\t', 0, 0);
    dmn_nodes = find( n_label(:,2)==3);
    check_roi = zeros(no_nodes, 2);
    check_roi( n_label(dmn_nodes,1), 1) = 1;
    
    
    load([data_path,'Ev_lan_nodes_12'], 'lan_node');
    check_roi( lan_node, 2) =1;
    
    no_net = 2;
elseif( yeo_flg==4)
    % load the construct nodes and combine them all
    load([data_path, 'construct6_network268_node_list'], 'check_roi');       
    manual_dec = 1; 
    % including bilateral hippocampus nodes
    if(manual_dec==1)
        hip_id = [93:97 230:234];
        check_roi(:, 3) = 0;
        check_roi(hip_id, 3) = 1;
    end
    
    sum_check = sum( check_roi,2);
    check_roi = (sum_check>0);
    no_net = 1;
end


% need to recover the lower triangle of the conn matrix
for run_idx = 1: no_runs
    for sub_id = 1: no_sub
        cur = squeeze(multi_run_all_mats(:,:,run_idx, sub_id));
        cur = cur+transpose(cur);
        multi_run_all_mats(:,:, run_idx, sub_id) = cur;
    end
end

lambda = [ 0 0.00001 0.0001 0.001 0.004 0.007 0.01 0.04 0.07 0.1 0.4 0.7 1 1.5 2 2.5 3 3.5 4 5 10 15 20];
lambda_no = length(lambda);

no_iter = 100;
scale_flg =0;
fold = 10;
bin_size = round( no_sub/fold);

error_flg = 0;


% select the scores to be combined
com_flg=2;

for ll = 1: lambda_no
    
    disp(['lambda = ', num2str(lambda(ll))]);
    cur_lambda = lambda(ll);
    
    if( yeo_flg <2)
        all_perf_r = zeros(no_iter, no_beh, no_net);
        all_perf_p = zeros(no_iter, no_beh, no_net);
    elseif( yeo_flg==2)
        all_perf_r = zeros(no_iter, no_beh);
        all_perf_p = zeros(no_iter, no_beh);
    elseif( yeo_flg>=3)
        all_perf_r = zeros(no_iter, no_beh, no_net);
        all_perf_p = zeros(no_iter, no_beh, no_net);
    end
    
    % combining the scores 
    % two options of combination based on clustering results
    % behavioral scores with similar prediction results are clustered together
    if( com_flg==1)
        comb_id = [1 2 13 16]; % the indices of the scores to be combined
    elseif( com_flg==2)
        comb_id = [3 5 11];
    elseif( com_flg==3)
        comb_id = [12 15];
    elseif( com_flg==4)
        comb_id = [4 6 7 10];
    elseif( com_flg==5)
        comb_id = [8 9];
    end
    
    comb_len = length( comb_id);
    
    for iter = 1: no_iter
        
        cur_order= randperm(no_sub); % shuffle the subject order
        cur_cov = cov_combined(cur_order, :);
               
        
        all_behav_res = datamat_res(:, comb_id);
        all_behav_orig = datamat_orig(:, comb_id);
        
        % first shuffle
        all_behav_res = all_behav_res(cur_order, :);
        all_behav_orig = all_behav_orig(cur_order, :);
        
        % then remove the NaN
        inner_va_id = find( sum(isnan(all_behav_orig),2)==0);
        cur_no_sub = length(inner_va_id);
        
        cur_behav_orig = all_behav_orig(inner_va_id, :);
        cur_behav_res = all_behav_res(inner_va_id, :);
        
        cur_cov_va = cur_cov(inner_va_id, :); % the 5 covariates
        
        va_order = cur_order(inner_va_id);
        
        % save the predicted values
        if( yeo_flg <2 || yeo_flg>2)
            behav_pred = zeros(cur_no_sub,no_net);
        else
            behav_pred = zeros(cur_no_sub, 1);
        end
        
        % using all runs
        all_mats = multi_run_all_mats(:,:,:, cur_order);
        cur_mat = all_mats(:,:, :, inner_va_id);
        cc = zeros(cur_no_sub, comb_len);
        
        for leftout = 1:fold;
            
            left_sub = (leftout-1)*bin_size+1: min(cur_no_sub, leftout*bin_size);
            
            % leave out subject from matrices and behavior
            
            train_mats = cur_mat;
            train_mats(:,:,:, left_sub) = [];
            % number of training subjects
            c_sub_no = size(train_mats, 4);
            
            % subject id included in the training set
            cur_train_sub = va_order;
            cur_train_sub(left_sub) = [];
            
            % regress the covariates within the training set
            train_cov = cur_cov_va;
            train_cov(left_sub, :) = [];
            
            train_behav_orig = cur_behav_orig;
            train_behav_orig(left_sub, :) = [];
            
            train_norm = zeros( size( train_behav_orig));
            
            test_res_left = cur_behav_res(left_sub, :);

            % regress the covariates for each score individually within the training
            for cb = 1: comb_len;
                [b, bint, train_behav_res] = regress( train_behav_orig(:, cb), [train_cov, ones(c_sub_no, 1)]);
                
                % normalize each single behavioral score after regression
                train_mu = mean(train_behav_res);
                train_std = std(train_behav_res);
                train_norm(:, cb) = (train_behav_res-train_mu)/train_std;

                % normalize the regressed scores of the test set using the parameters from the training
                cc(left_sub, cb) = (test_res_left(:,cb) -train_mu)/train_std;
                
                clear train_mu train_std;
            end
            
            train_norm_sum = sum( train_norm, 2); % the combined score after regression and normalization
            
            % matrices of the test subjects
            test_mat = cur_mat(:,:,:, left_sub);
            % for each construct compute the positive sum of edges
            left_no = length( left_sub);
            
            if( yeo_flg <2 || yeo_flg>2)
                for con = 1: no_net;
                    if( yeo_flg ==0)
                        cur_con_nodes = find( check_roi(:, con)>0);
                    elseif( yeo_flg ==1)
                        cur_con_nodes = find( check_roi(:, con)==1);
                    elseif( yeo_flg== 3)
                        cur_con_nodes = find( check_roi(:, con)==1);
                    elseif( yeo_flg==4)
                        cur_con_nodes = find( check_roi(:, con)==1);
                    end
                    cur_no = length( cur_con_nodes);
                    
                    all_nodes = ones(1, no_nodes);
                    all_nodes(cur_con_nodes) = 0;
                    all_d_nodes = find( all_nodes==1);
                    aa = ones(no_nodes, no_nodes);
                    aa_upp = triu(aa, 1);
                    aa_upp(all_d_nodes, all_d_nodes) = 0;
                    upp_id = find( aa_upp);
                    
                    cur_con_mat = [];
                    cur_con_test = [];
                    
                    for run_idx = 1:no_runs
                        cur_train_mat = reshape(squeeze(train_mats(:,:,run_idx,:)), no_nodes*no_nodes, c_sub_no);
                        cur_con_mat = [cur_con_mat; cur_train_mat(upp_id,:)];
                        
                        cur_test_mat = reshape( squeeze(test_mat(:,:,run_idx,:)), no_nodes*no_nodes, left_no);
                        cur_con_test = [cur_con_test; cur_test_mat(upp_id,:)];
                    end
                    
                    if( scale_flg==0)
                        scale = 0.2;
                    else
                        scale = 1;
                    end
                    
                    
                    [pred, estimate] = kernel_prediction(cur_con_mat, cur_con_test, train_norm_sum, cur_lambda, 'Gaussian', scale);
                    behav_pred(left_sub, con) = pred;
                end
            elseif( yeo_flg ==2)
                cur_con_mat =[];
                cur_con_test = [];
                
                for run_idx = 1:no_runs
                    cur_train_mat = reshape(squeeze(train_mats(:,:,run_idx,:)), no_nodes*no_nodes, c_sub_no);
                    cur_con_mat = [cur_con_mat; cur_train_mat(upp_id,:)];
                    
                    cur_test_mat = reshape( squeeze(test_mat(:,:,run_idx,:)), no_nodes*no_nodes, left_no);
                    cur_con_test = [cur_con_test; cur_test_mat(upp_id,:)];
                end
                
                if( scale_flg==0)
                    scale = 0.2;
                else
                    scale = 1;
                end
                
                [pred, estimate] = kernel_prediction(cur_con_mat, cur_con_test, train_norm_sum, cur_lambda, 'Gaussian', scale);
                behav_pred(left_sub) = pred;
            end
            
        end
        
        if(yeo_flg<2 || yeo_flg>2)
            
            for con= 1:no_net;
                [R, P] = corr( squeeze(behav_pred(:, con)), sum( cc, 2));
                all_perf_r(iter,  1, con) = R;
                all_perf_p(iter , 1, con) = P;
                
            end
        else
            
            [R, P] = corr( behav_pred, sum(cc, 2));
            all_perf_r(iter, 1) = R;
            all_perf_p(iter, 1) = P;
            
        end
        
        
    end
    
    
    if( yeo_flg ==0)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_singlecon_KRR_10fold_normalized_allruns_set227_combinedscore_set', num2str(com_flg), '_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
        end
        
    elseif( yeo_flg==1)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_yeonet_KRR_10fold_normalized_allruns_set227_combinedscore_set', num2str(com_flg), '_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
            
        end
    elseif( yeo_flg==2)
        if( scale_flg ==0)
            if( nan_flg==1)               
                save([save_path, 'trans_data_wholebrain_KRR_10fold_normalized_allruns_set227_combinedscore_set', num2str(com_flg), '_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
        end
        
    elseif( yeo_flg==3)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_dmnEv_KRR_10fold_normalized_allruns_set227_combinedscore_set', num2str(com_flg), '_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
                
            end
            
        end
    elseif( yeo_flg==4)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_all6con_KRR_10fold_normalized_allruns_set227_combinedscore_set', num2str(com_flg), '_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
                
            end
            
        end
    end
end
