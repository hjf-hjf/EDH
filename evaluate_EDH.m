function evaluation_info=evaluate_EDH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,param)
      
    tic;
    
    % Hash codes learning
    [B] = solveEDH(XKTrain', YKTrain', LTrain', param);
    
    % Hash functions learning
    L_T =LTrain';
    for i = 1:size(L_T,2)
        L_norm(:,i) = L_T(:,i)/norm(L_T(:,i),2);
    end
    belta = param.belta;
    bits = param.nbits;
%     XW = (XKTrain'*XKTrain+belta*eye(size(XKTrain,2)))    \    (XKTrain'*B');
%     YW = (YKTrain'*YKTrain+belta*eye(size(YKTrain,2)))    \    (YKTrain'*B');

    XW_temp = ((B * B' + belta * eye(size(B,1))) \ (bits * B * L_norm' * L_norm * XKTrain + belta * B * XKTrain)) / (XKTrain' * XKTrain + 0.01 * eye(size(XKTrain,2)));
    YW_temp = ((B * B' + belta * eye(size(B,1))) \ (bits * B * L_norm' * L_norm * YKTrain + belta * B * YKTrain)) / (YKTrain' * YKTrain + 0.01 * eye(size(YKTrain,2)));
    XW = XW_temp';
    YW = YW_temp';
     
    traintime=toc;
    evaluation_info.trainT=traintime;
    
    tic;
    
    % Cross-Modal Retrieval
    BxTest = compactbit(sign(XKTest*XW)>0);
%     BxTest = compactbit(param.mu*(XKTest*(pinv(U1))'*R1'+LTest*(pinv(W))'*R2')>0);
    BxTrain = compactbit(B'>0);
    DHamm = hammingDist(BxTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    
    ByTest = compactbit(sign(YKTest*YW)> 0);
%     ByTest = compactbit(param.mu*(YKTest*(pinv(U2))'*R1'+LTest*(pinv(W))'*R2')> 0);
    ByTrain = compactbit(B'>0); % ByTrain = BxTrain;
    DHamm = hammingDist(ByTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    compressiontime=toc;
    
    evaluation_info.compressT=compressiontime;
    %evaluation_info.BxTrain = BxTrain;
    %evaluation_info.ByTrain = ByTrain;
    %evaluation_info.B = B;

end
