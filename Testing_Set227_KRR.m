% This is an example to show how to run the kernel ridge regression model 
% to predict behavioral scores based on the connectivity matrices. 
% We provided a subset of data that was collected and processed at Yale MRRC. 
% The provided dataset includes connectivity matrices, behavioral scores, and 
% covariates from all subjects.

clear
clc

% % define your data path
% data_path = '';
% % define your path to save the results
% save_path = '';
% % define your path to save the prediction errors
% error_path = '';


% load the matrix after the regression
% Each edge in the matrix is the residue after regressing out the covariates
load([data_path, 'trans_data_set227_mat_cov_regressed'], 'res_mat');

% load the original scores
load([data_path, 'trans_data_all_scores_set227'],'all_score');
% 146 = 124 + 16 + 6; other sub-scores + cognitive_scores + clinical scores
% the first 124 columns are sub-scores
% column 125-140 are the 16 cognitive scores
% column 141-146 are the 6 clinical scores

% load the regressed scores (for evaluation purpose)
% the covariates are regressed from the behavioral scores 
load([data_path, 'trans_data_set227_allscores_cov_regressed'], 'all_res');

% load the covariates
load([data_path, 'trans_data_combined_update_cov_diag_set227'], 'cov_combined');


% choose the set of behavioral scores to predict
nan_flg = 1;
if( nan_flg ==1)
    % cognitive scores (16)
    datamat_orig = all_score(:, 125:140);
    datamat_res = all_res(:, 125:140);
elseif( nan_flg==2)
    % all rest of the sub scores (124)
    datamat_orig = all_score(:, 1:124);
    datamat_res = all_res(:, 1:124);
elseif( nan_flg ==3)
    % clinical scores (6)
    datamat_orig = all_score(:, 141:146);
    datamat_res = all_res(:, 141: 146);
elseif( nan_flg ==4)
    % load the newly computed brief composite score
    % we selected a subset of the brief subscore and create a new composite score
    load([data_path, 'trans_data_set227_brief_new_composite'], 'new_brief_orig', 'new_brief_res', 'new_name');

    datamat_orig = new_brief_orig;
    datamat_res = new_brief_res;
end

% find the number of behavioral scores to predict
no_beh = size( datamat_res, 2);

% matrices after regression
multi_run_all_mats = res_mat;
% the connectivity matrices are organized as a 4D matrix of no_of_nodes * no_of_nodes * no_runs * no_subjects
no_runs = size(multi_run_all_mats, 3);  % each subject has 8 runs
no_sub = size(multi_run_all_mats, 4);
no_nodes = size(multi_run_all_mats,1); 
% the atlas we used for creating the connectivity matrices has 268 nodes.

% find the indices of the edges in the upper triangle of a matrix
% the connectivity matrix is symmetric
aa = ones( no_nodes, no_nodes);
aa_upp = triu(aa,1);
upp_id = find( aa_upp>0);
upp_len = length(upp_id);

% To select edges from pre-defined networks/constructs
% The prediction is based on an individual network/construct
% yeo_flg = 1; % the yeo network (7 networks)
% yeo_flg = 3; % use DMN and Ev language
% yeo_flg = 2; % use the whole brain
yeo_flg = 0; % use the constructs (6 constructs defined based on NeuroSynth: attention, perception, declarative memory, language, cognitive control and working memory) 
% yeo_flg = 4; % combine all 6 constructs;

if( yeo_flg==0)
    % load the construct nodes    
    load([data_path, 'construct6_network268_node_list'], 'check_roi');
    % 'check_roi' was created based on NeuroSynth results
    % it is a labeling matrix where each column represents one construct
    % and each row represents one ROI
    % the order of the columns: attention, perception, declarative memory, language, cognitive control, working memory
        
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
    % load the Yeo's 7 network definition
    
    yeomap = dlmread([data_path, 'Parc268toYeo7netlabel']);
    
    no_net = size( yeomap, 2);
    check_roi = yeomap;
elseif( yeo_flg ==2)
    % use the whole brain;
elseif( yeo_flg ==3)
    % use the dmn and Ev's language network
    file_name = 'Shen268_10network';
    % use the dmn network definition from the in-house developed resting state functional brain network
    n_label = dlmread([data_path, file_name], '\t', 0, 0);
    dmn_nodes = find( n_label(:,2)==3);
    check_roi = zeros(no_nodes, 2);
    check_roi( n_label(dmn_nodes,1), 1) = 1;
    
    % use the language network: 
    % Lipkin, Ben (2022). LanA Dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.20425209.v1
    load([data_path,'Ev_lan_nodes_12'], 'lan_node');
    check_roi( lan_node, 2) =1;
    
    no_net = 2;
elseif( yeo_flg==4)
    % load the construct nodes and combine all of them   
    load([data_path, 'construct6_network268_node_list'], 'check_roi');       
    manual_dec = 1; % 
    if(manual_dec==1)
        hip_id = [93:97 230:234];
        check_roi(:, 3) = 0;
        check_roi(hip_id, 3) = 1;
    end
    
    sum_check = sum( check_roi,2);
    check_roi = (sum_check>0);
    no_net = 1;
end

% filling the lower triangle of the conn matrix
for run_idx = 1: no_runs
    for sub_id = 1: no_sub
        cur = squeeze(multi_run_all_mats(:,:,run_idx, sub_id));
        cur = cur+transpose(cur);
        multi_run_all_mats(:,:, run_idx, sub_id) = cur;
    end
end

% lambda is the hyper-parameter of the kernle ridge regression
lambda = [ 0 0.00001 0.0001 0.001 0.004 0.007 0.01 0.04 0.07 0.1 0.4 0.7 1 1.5 2 2.5 3 3.5 4 5 10 15 20];
lambda_no = length(lambda);

no_iter = 1; % number of iteration for the k-fold cross validation
scale_flg =0; % scale of the kernel 
fold = 10; % k-fold
bin_size = round( no_sub/fold); % size of each fold

error_flg = 0; % whether or not to save the prediction error

for ll = 1: lambda_no % loop through all lambda values
    
    disp(['lambda = ', num2str(lambda(ll))]);
    cur_lambda = lambda(ll);

    % create placeholders for prediction performance
    if( yeo_flg <2)
        all_perf_r = zeros(no_iter, no_beh, no_net);
        all_perf_p = zeros(no_iter, no_beh, no_net);
    elseif( yeo_flg==2)
        all_perf_r = zeros(no_iter, no_beh);
        all_perf_p = zeros(no_iter, no_beh);
    elseif( yeo_flg==3)
        all_perf_r = zeros(no_iter, no_beh, no_net);
        all_perf_p = zeros(no_iter, no_beh, no_net);
    end
    
    % start the prediction
    for iter = 1: no_iter
        
        cur_order= randperm(no_sub); % shuffle the subject order
        cur_cov = cov_combined(cur_order, :); % shuffle the covariates accordingly

        % predict a single behavioral score
        for behav_idx = 1: no_beh;
                        
            % select the behavioral score to predict           
            all_behav_res = datamat_res(:, behav_idx);
            all_behav_orig = datamat_orig(:, behav_idx);
            
            % first shuffle the behavioral score 
            all_behav_res = all_behav_res(cur_order);
            all_behav_orig = all_behav_orig(cur_order);
            
            % then remove subjects with NaN scores
            inner_va_id = find( isnan(all_behav_orig)==0);
            cur_no_sub = length(inner_va_id);

            % all with valid scores
            cur_behav_orig = all_behav_orig(inner_va_id);
            cur_behav_res = all_behav_res(inner_va_id);
            
            cur_cov_va = cur_cov(inner_va_id, :); % the 5 covariates            
            va_order = cur_order(inner_va_id);
            
            % placeholders to save the predicted values
            if( yeo_flg <2 || yeo_flg>2)
                behav_pred = zeros(cur_no_sub,no_net);
            else 
                behav_pred = zeros(cur_no_sub, 1);
            end
            
            % using all runs
            all_mats = multi_run_all_mats(:,:,:, cur_order);
            cur_mat = all_mats(:,:, :, inner_va_id);

            % start the cross validation loop
            for leftout = 1:fold;
                
                left_sub = (leftout-1)*bin_size+1: min(cur_no_sub, leftout*bin_size);
                
                % leave out subjects from matrices and behavior
                
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
                train_behav_orig(left_sub) = [];
                
                [b, bint, train_behav_res] = regress( train_behav_orig, [train_cov, ones(c_sub_no, 1)]);

                % normalize the score after regression
                train_mu = mean(train_behav_res);
                train_std = std(train_behav_res);
                train_norm = (train_behav_res-train_mu)/train_std;
                
                % matrices of the test subjects
                test_mat = cur_mat(:,:,:, left_sub);
                
                left_no = length( left_sub);
                
                if( yeo_flg <2 || yeo_flg>2)
                    for con = 1: no_net;
                        % find the nodes for the selected network/construct
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

                        % find the edges between the selected nodes and the rest of the brain                
                        all_nodes = ones(1, no_nodes);
                        all_nodes(cur_con_nodes) = 0;
                        all_d_nodes = find( all_nodes==1);
                        aa = ones(no_nodes, no_nodes);
                        aa_upp = triu(aa, 1);
                        aa_upp(all_d_nodes, all_d_nodes) = 0;
                        upp_id = find( aa_upp);
                        
                        cur_con_mat = [];
                        cur_con_test = [];

                        % running the prediction from each individual runs
                        for run_idx = 1:no_runs
                            cur_train_mat = reshape(squeeze(train_mats(:,:,run_idx,:)), no_nodes*no_nodes, c_sub_no);
                            cur_con_mat = [cur_con_mat; cur_train_mat(upp_id,:)];
                            
                            cur_test_mat = reshape( squeeze(test_mat(:,:,run_idx,:)), no_nodes*no_nodes, left_no);
                            cur_con_test = [cur_con_test; cur_test_mat(upp_id,:)];
                        end

                        % scale of the Gaussian kernel
                        if( scale_flg==0)
                            scale = 0.2;
                        else
                            scale = 1;
                        end
                        
                        % running the kernel ridge regression
                        [pred, estimate] = kernel_prediction(cur_con_mat, cur_con_test, train_norm, cur_lambda, 'Gaussian', scale);                        
                        behav_pred(left_sub, con) = pred;                        
                    end
                elseif( yeo_flg ==2)
                    cur_con_mat =[];
                    cur_con_test = [];

                    % concatenate the edges from all 8 runs
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
                                        
                    [pred, estimate] = kernel_prediction(cur_con_mat, cur_con_test, train_norm, cur_lambda, 'Gaussian', scale);                    
                    behav_pred(left_sub) = pred;
                end
                
            end
            
            if(yeo_flg<2 || yeo_flg>2)
                pred_oo = zeros(no_sub, no_net); % prediction before the shuffle
                error_oo = zeros(no_sub, no_net); % error before the shuffle
                for con= 1:no_net;
                    % compute the correlation between the predicted scores and the observed scores (after regression)
                    [R, P] = corr( squeeze(behav_pred(:, con)), cur_behav_res);
                    all_perf_r(iter,  behav_idx, con) = R;
                    all_perf_p(iter , behav_idx, con) = P;
                    
                    % compute the prediction error
                    pred_ss = zeros(1, no_sub); % prediction after the shuffle
                    pred_ss(inner_va_id) = behav_pred(:,con);
                    % prediction error in the original order of subjects                   
                    pred_oo(cur_order, con) = pred_ss;
                    
                    error_ss = zeros(1, no_sub);
                    cc_mean = mean(cur_behav_res);
                    cur_behav_mean = cur_behav_res-cc_mean;
                    cc_norm = norm(cur_behav_mean); % normalized the scores (after regression)
                    error_ss(inner_va_id) = (behav_pred(:, con) - cur_behav_mean/cc_norm); % signed error                    
                    error_oo(cur_order, con) = error_ss;
                    
                    clear cc_norm cc_mean;
                end
            else
                pred_oo = zeros(no_sub, 1);
                error_oo = zeros(no_sub, 1);

                % compute the correlation between the predicted scores and the observed scores (after regression)
                [R, P] = corr( behav_pred, cur_behav_res);
                all_perf_r(iter, behav_idx) = R;
                all_perf_p(iter, behav_idx) = P;
                
                % compute the prediction error
                pred_ss = zeros( no_sub, 1); % prediction after the shuffle
                pred_ss(inner_va_id) = behav_pred;
                % in the original order
                pred_oo(cur_order) = pred_ss;
                
                error_ss = zeros(no_sub, 1);
                cc_mean = mean(cur_behav_res);
                cur_behav_mean = cur_behav_res-cc_mean;
                cc_norm = norm(cur_behav_mean);
                error_ss(inner_va_id) = (behav_pred - cur_behav_mean/cc_norm); % signed error
                error_oo(cur_order) = error_ss;
                
                clear cc_norm cc_mean;
            end

            % save the prediction errors
            if( error_flg ==1)
                error_path = '/data22/mri_group/xilin_data/alex/trans_data/set227/errors/';
                if( yeo_flg==0)
                    if( nan_flg ==1)
                        save([error_path,'trans_data_set227_KRR_error_singlecon_cogbehav', num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg ==2)
                        save([error_path,'trans_data_set227_KRR_error_singlecon_subscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg==3)
                        save([error_path,'trans_data_set227_KRR_error_singlecon_clicscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    end
                elseif( yeo_flg ==1)
                    if( nan_flg ==1)
                        save([error_path,'trans_data_set227_KRR_error_yeonet_cogbehav', num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg ==2)
                        save([error_path,'trans_data_set227_KRR_error_yeonet_subscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg==3)
                        save([error_path,'trans_data_set227_KRR_error_yeonet_clicscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    end
                elseif( yeo_flg==2)
                    if( nan_flg ==1)
                        save([error_path,'trans_data_set227_KRR_error_wholebrain_cogbehav', num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg ==2)
                        save([error_path,'trans_data_set227_KRR_error_wholebrain_subscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg==3)
                        save([error_path,'trans_data_set227_KRR_error_wholebrain_clicscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    end
                elseif( yeo_flg==3)
                    if( nan_flg ==1)
                        save([error_path,'trans_data_set227_KRR_error_dmnEv_cogbehav', num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg ==2)
                        save([error_path,'trans_data_set227_KRR_error_dmnEv_subscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg==3)
                        save([error_path,'trans_data_set227_KRR_error_dmnEv_clicscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    end
                elseif( yeo_flg ==4)
                    if( nan_flg ==1)
                        save([error_path,'trans_data_set227_KRR_error_all6con_cogbehav', num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg ==2)
                        save([error_path,'trans_data_set227_KRR_error_all6con_subscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    elseif( nan_flg==3)
                        save([error_path,'trans_data_set227_KRR_error_all6con_clicscores',num2str(behav_idx), '_iter', num2str(iter)], 'error_oo', 'pred_oo', 'cur_order');
                    end
                end
            end
        end
    end
    % save the prediction performance
    if( yeo_flg ==0)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_singlecon_KRR_10fold_normalized_allruns_set227_cogbehav_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==2)
                save([save_path, 'trans_data_singlecon_KRR_10fold_normalized_allruns_set227_subscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==3)
                save([save_path, 'trans_data_singlecon_KRR_10fold_normalized_allruns_set227_cliscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==4)
                save([save_path, 'trans_data_singlecon_KRR_10fold_normalized_allruns_set227_briefcomp_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
            
        end
        
    elseif( yeo_flg==1)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_yeonet_KRR_10fold_normalized_allruns_set227_cogbehav_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==2)
                save([save_path, 'trans_data_yeonet_KRR_10fold_normalized_allruns_set227_subscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==3)
                save([save_path, 'trans_data_yeonet_KRR_10fold_normalized_allruns_set227_clicscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==4)
                save([save_path, 'trans_data_yeonet_KRR_10fold_normalized_allruns_set227_briefcomp_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
            
        end
    elseif( yeo_flg==2)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_wholebrain_KRR_10fold_normalized_allruns_set227_cogbehav_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg ==2)
                save([save_path, 'trans_data_wholebrain_KRR_10fold_normalized_allruns_set227_subscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==3)
                save([save_path, 'trans_data_wholebrain_KRR_10fold_normalized_allruns_set227_clicscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==4)
                save([save_path, 'trans_data_wholebrain_KRR_10fold_normalized_allruns_set227_briefcomp_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end            
        end
        
    elseif( yeo_flg==3)
        if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_dmnEv_KRR_10fold_normalized_allruns_set227_cogbehav_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg ==2)
                save([save_path, 'trans_data_dmnEv_KRR_10fold_normalized_allruns_set227_subscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==3)
                save([save_path, 'trans_data_dmnEv_KRR_10fold_normalized_allruns_set227_clicscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==4)
                save([save_path, 'trans_data_dmnEv_KRR_10fold_normalized_allruns_set227_briefcomp_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
            
        end
        
    elseif( yeo_flg==4)
         if( scale_flg ==0)
            if( nan_flg==1)
                save([save_path, 'trans_data_all6con_KRR_10fold_normalized_allruns_set227_cogbehav_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg ==2)
                save([save_path, 'trans_data_all6con_KRR_10fold_normalized_allruns_set227_subscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==3)
                save([save_path, 'trans_data_all6con_KRR_10fold_normalized_allruns_set227_clicscores_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            elseif( nan_flg==4)
                save([save_path, 'trans_data_all6con_KRR_10fold_normalized_allruns_set227_briefcomp_lambda', num2str(ll)], 'all_perf_r', 'cur_lambda');
            end
            
        end
    end
    
    
end

