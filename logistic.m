function test_ans=logistic()
    clear;
    Test = load ('test.txt');
    Train = load('train.txt');
    [row,col] = size(Train);
    Train = Train(randperm(row), :);
    val = Train( 1: 2000 , : );
    train = Train( 2001 : row, : );

    %train=Train;

    K=10;
    iter = 5000;
    step = 0.5;
    delta = 0.9996;
    lambda = 600;
    [w,step, Accuracy_train, Recall_train, Precision_train, F1_train,  Accuracy_val, Recall_val, Precision_val, F1_val] = logistic_train(train, val, iter, step, delta,lambda);
    %K�۽�����֤
    %[Accuracy_train, Recall_train, Precision_train, F1_train,  Accuracy_val, Recall_val, Precision_val, F1_val] = K_Fold(K,Train, iter, step, delta,lambda);
    % fprintf('final train:\n Accuracy: %.6f  Recall: %.6f  Precision: %.6f  F1: %.6f\n',Accuracy_train(iter),Recall_train(iter),Precision_train(iter),F1_train(iter))
    % fprintf('final val:\n Accuracy: %.6f  Recall: %.6f  Precision: %.6f  F1: %.6f\n',Accuracy_val(iter),Recall_val(iter),Precision_val(iter),F1_val(iter))
    %����
    test_ans=logistic_test(w,Test);
    xlswrite('final_ans.xlsx',test_ans);

    %%%%%%%%%%%%%��ͼ%%%%%%%%%%%%%%
    figure;
    subplot(2, 2, 1);
    plot(1:iter, Accuracy_train, 'b');hold on
    plot(1:iter, Accuracy_val, 'r');hold off
    title('RL׼ȷ��');xlabel('��������'),ylabel('׼ȷ��');legend('ѵ����','��֤��');grid on;
    subplot(2, 2, 2);
    plot(1:iter, F1_train, 'b');hold on
    plot(1:iter, F1_val, 'r');hold off
    title('RL F1');xlabel('��������'),ylabel('F1');legend('ѵ����','��֤��');grid on;
    subplot(2, 2, 3);
    plot(1:iter, Recall_train, 'b');hold on
    plot(1:iter, Recall_val, 'r');hold off
    title('RL �ٻ���');xlabel('��������'),ylabel('�ٻ���');legend('ѵ����','��֤��');grid on;
    subplot(2, 2, 4);
    plot(1:iter, Precision_train, 'b');hold on
    plot(1:iter, Precision_val, 'r');hold off
    title('RL ��ȷ��');xlabel('��������'),ylabel('��ȷ��');legend('ѵ����','��֤��');grid on;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
function [w,step, Accuracy_train, Recall_train, Precision_train, F1_train,  Accuracy_val, Recall_val, Precision_val, F1_val]= logistic_train(train, val, iter, step, delta , lambda)
    %���ݼ�����
    [train_row,train_col] = size(train);
    tmp_train = [ones(train_row, 1) ,train( : , 1 : train_col - 1)];
    sign_train = train( : , train_col);
    [val_row,val_col] = size(val);
    tmp_val = [ones(val_row,1), val( : , 1 : val_col - 1)];
    sign_val = val( : , val_col);
    %��ʼ��
    TP_train = zeros(1,iter); 
    FN_train = zeros(1,iter);
    TN_train = zeros(1,iter);
    FP_train = zeros(1,iter);
    TP_val = zeros(1,iter);
    FN_val = zeros(1,iter);
    TN_val = zeros(1,iter);
    FP_val = zeros(1,iter);
    
    w = ones(1, train_col) / train_col;%���ó�ʼW
   
    for i = 1 : iter %����
        %������ʧ����
        Err = ((1 ./ (1 + exp(-w * tmp_train')) - sign_train') * tmp_train + lambda * w) / train_row;
%         %����ݶ��½���
%         pos=randi([1 train_row]);
%         Err = ((1 / (1 + exp(-w * tmp_train(pos, : )')) - sign_train(pos)') * tmp_train(pos, : ) + lambda * w) / train_row;
        Err(abs(Err)<eps)=0;
        if abs(Err) < eps
            break;
        end
        step = step * delta;%������̬����
        w = w - step * Err;%����w
        
        train_ans = 1 ./ (1 + exp(-w * tmp_train'))';
        train_ans(train_ans >= 0.5) = 1;
        train_ans(train_ans < 0.5) = 0;
        S = train_ans + 10 * sign_train;
        TP_train(i) = sum(S( : ) == 11);
        FN_train(i) = sum(S( : ) == 10);
        TN_train(i) = sum(S( : ) == 0);
        FP_train(i) = sum(S( : ) == 1);
        
        val_ans = 1 ./ (1 + exp(-w * tmp_val'))';
        val_ans(val_ans >= 0.5) = 1;
        val_ans(val_ans < 0.5) = 0;
        S = val_ans + 10 * sign_val;
        TP_val(i) = sum(S( : ) == 11);
        FN_val(i) = sum(S( : ) == 10);
        TN_val(i) = sum(S( : ) == 0);
        FP_val(i) = sum(S( : ) == 1);
    end
    Accuracy_train = (TP_train + TN_train) ./ (TP_train + FP_train + TN_train + FN_train);
    Recall_train  = TP_train ./ (TP_train + FN_train);
    Precision_train  = TP_train ./ (TP_train + FP_train);
    F1_train = (2 *  Precision_train  .* Recall_train) ./ (Precision_train  + Recall_train); 
    
    Accuracy_val = (TP_val + TN_val) ./ (TP_val + FP_val + TN_val + FN_val);
    Recall_val  = TP_val ./ (TP_val + FN_val);
    Precision_val  = TP_val ./ (TP_val + FP_val);
    F1_val = (2 *  Precision_val  .* Recall_val) ./ (Precision_val  + Recall_val); 
    
end
function test_ans=logistic_test(w,test)%�о�����
    [test_row,test_col] = size(test);
    tmp_test = [ones(test_row,1), test( : , 1 : test_col - 1)];
    test_ans = 1 ./ (1 + exp(-w * tmp_test'))';
    test_ans(test_ans >= 0.5) = 1;
    test_ans(test_ans < 0.5) = 0;
end

function [Acc_train, Re_train, Pre_train, F_train,  Acc_val, Re_val, Pre_val, F_val] =K_Fold(K, Train, iter, step, delta,lambda)
    Indices = crossvalind('Kfold', size(Train,1), K);
    Acc_train = zeros(1,iter);
    Re_train = zeros(1,iter);
    Pre_train = zeros(1,iter);
    F_train = zeros(1,iter);
    Acc_val = zeros(1,iter);
    Re_val = zeros(1,iter);
    Pre_val = zeros(1,iter);
    F_val = zeros(1,iter);

    for i=1:K %K�۽�����֤
        val = Train((Indices == i),: );
        train = Train((Indices ~= i),: );
        [~, Accuracy_train, Recall_train, Precision_train, F1_train,  Accuracy_val, Recall_val, Precision_val, F1_val] = logistic_train(train, val, iter, step, delta,lambda);
        %��10��ƽ��
        Acc_train = Acc_train + Accuracy_train / K;
        Re_train = Re_train + Recall_train / K ;
        Pre_train = Pre_train + Precision_train / K;
        F_train = F_train + F1_train / K;
        Acc_val = Acc_val + Accuracy_val/ K;
        Re_val = Re_val + Recall_val / 10 ;
        Pre_val = Pre_val + Precision_val / K;
        F_val = F_val + F1_val / K;
    end
end
