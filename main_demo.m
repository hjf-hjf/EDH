close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);  
end

db = {'mirflickr25k','nusData','IAPRTC-12'};    %'mirflickr25k','nusData','IAPRTC-12'       
hashmethods = {'EDH'};                       
loopnbits = [16,32,64,128];

param.top_K = 1000; 

for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    
    diary(['./results/conv_',db_name,'_result.txt']);
    diary on;
    
    %% load dataset
    load(['./',db_name,'.mat']);
    result_name = [result_URL 'final_' db_name '_result' '.mat'];

    if strcmp(db_name, 'mirflickr25k')
        inx = randperm(size(L_tr,1),size(L_tr,1));
        XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
        XTest = I_te; YTest = T_te; LTest = L_te;
        
   elseif strcmp(db_name, 'nusData')
        inx = randperm(size(L_tr,1),20000);
        XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
        XTest = I_te; YTest = T_te; LTest = L_te;

    elseif strcmp(db_name, 'IAPRTC-12')
        inx = randperm(size(L_tr,1),size(L_tr,1));
        XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
        XTest = I_te; YTest = T_te; LTest = L_te;

    end
    %% Kernel representation
    [n, ~] = size(YTrain);
    if strcmp(db_name, 'mirflickr25k')
        n_anchors = 2000;
        anchor_image = XTrain(randsample(n, n_anchors),:); 
        anchor_text = YTrain(randsample(n, n_anchors),:);
        XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
        YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
        
   elseif strcmp(db_name, 'nusData')
        n_anchors = 2000;
        anchor_image = XTrain(randsample(n, n_anchors),:); 
        anchor_text = YTrain(randsample(n, n_anchors),:);
        XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
        YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
             
    elseif strcmp(db_name, 'IAPRTC-12')
        n_anchors = 2000;
        anchor_image = XTrain(randsample(n, n_anchors),:); 
        anchor_text = YTrain(randsample(n, n_anchors),:);
        XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
        YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
        
    end
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        
        for jj = 1:length(hashmethods)
            switch(hashmethods{jj})
                case 'EDH'
                    fprintf('......%s start...... \n\n', 'EDH');
                    EDHparam = param;
                    EDHparam.lambda = 0.5; EDHparam.mu = 0.5; EDHparam.gamma = 1e2;
                    EDHparam.alpha = 1e4; EDHparam.belta = 1e4; EDHparam.maxIter = 6;
                    
                    if strcmp(db_name, 'IAPRTC-12')
                        EDHXKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1));
                        EDHXKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
                        EDHYKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1));
                        EDHYKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
                        
                    elseif strcmp(db_name, 'mirflickr25k')
                        EDHXKTest = XKTest;
                        EDHXKTrain = XKTrain;
                        EDHYKTest = YKTest;
                        EDHYKTrain = YKTrain;
                        
                    elseif strcmp(db_name, 'nusData')
                        EDHXKTest = XKTest;
                        EDHXKTrain = XKTrain;
                        EDHYKTest = YKTest;
                        EDHYKTrain = YKTrain;
                        
                    end
                    eva_info_ = evaluate_EDH(EDHXKTrain,EDHYKTrain,LTrain,EDHXKTest,EDHYKTest,LTest,EDHparam);                
            end
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end  
    %% Results
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            % MAP
            Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;
            
            %NDCG   shuffle training and test data
%             Image_VS_Text_NDCG{jj,ii} = eva_info{jj,ii}.Image_VS_Text_NDCG;
%             Text_VS_Image_NDCG{jj,ii} = eva_info{jj,ii}.Text_VS_Image_NDCG;      
            
            % Precision VS Recall
            Image_VS_Text_recall{jj,ii,:}    = eva_info{jj,ii}.Image_VS_Text_recall';
            Image_VS_Text_precision{jj,ii,:} = eva_info{jj,ii}.Image_VS_Text_precision';
            Text_VS_Image_recall{jj,ii,:}    = eva_info{jj,ii}.Text_VS_Image_recall';
            Text_VS_Image_precision{jj,ii,:} = eva_info{jj,ii}.Text_VS_Image_precision';

            % Top number Precision
            Image_To_Text_Precision{jj,ii,:} = eva_info{jj,ii}.Image_To_Text_Precision;
            Text_To_Image_Precision{jj,ii,:} = eva_info{jj,ii}.Text_To_Image_Precision;
            
            % Time
            trainT{jj,ii} = eva_info{jj,ii}.trainT;
            testT{jj,ii} = eva_info{jj,ii}.compressT;
        end
    end
  
    save(result_name,'eva_info','EDHparam','loopnbits','hashmethods',...
        'trainT','testT','Image_VS_Text_MAP','Text_VS_Image_MAP','Image_VS_Text_recall','Image_VS_Text_precision',...
        'Text_VS_Image_recall','Text_VS_Image_precision','Image_To_Text_Precision','Text_To_Image_Precision','-v7.3');
    
%     save(result_name,'eva_info','EDHparam','loopnbits','hashmethods',...
%         'trainT','testT','Image_VS_Text_MAP','Text_VS_Image_MAP','Image_VS_Text_recall','Image_VS_Text_precision',...
%         'Text_VS_Image_recall','Text_VS_Image_precision','Image_VS_Text_NDCG','Text_VS_Image_NDCG','Image_To_Text_Precision','Text_To_Image_Precision','-v7.3');
    
    diary off;
end
