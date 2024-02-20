clear;clc;
num_train = 60000;
num_test = 10000;
[imgs_train,labels_train] = readMNIST('training set/train-images.idx3-ubyte','training set/train-labels.idx1-ubyte',num_train,0);
[imgs_test,labels_test] = readMNIST('test set/t10k-images.idx3-ubyte','test set/t10k-labels.idx1-ubyte',num_test,0);

t = zeros(10,num_train);
for i = 1: num_train
    t(labels_train(i)+1,i) = 1;
end

imgs_bias_train = [imgs_train, ones(num_train,1)];
imgs_bias_test = [imgs_test, ones(num_test,1)];

lr_sig = 0.01;
lr_ReLU = 2e-3;
H = [10,20,50];

epoch = 20;
steps = 100;
% total PoE = epoch * num_train / steps
error_train_sig = zeros(3,epoch*num_train/steps);
error_test_sig = zeros(3,epoch*num_train/steps);

error_train_ReLU = zeros(3,epoch*num_train/steps);
error_test_ReLU = zeros(3,epoch*num_train/steps);

for h = 1:3
    w_sig = randn(size(imgs_bias_train,2),H(h));
    v_sig = randn(H(h),10);
    w_ReLU = w_sig;
    v_ReLU = v_sig;
    for e = 1:epoch
      for i = 1:num_train
          % sigmoid
          g_sig = imgs_bias_train(i,:) * w_sig;
          y_sig = logsig(g_sig);
          u_sig = y_sig*v_sig;
          z_sig = softmax(u_sig');
          % derivative of logsig y = y.*(1-y)
          grad_w_sig =((y_sig.*(1-y_sig))'.* (v_sig*(t(:,i) - z_sig)))* imgs_bias_train(i,:);
          grad_v_sig = (t(:,i) - z_sig) * y_sig;
          w_sig = w_sig + lr_sig*grad_w_sig';
          v_sig = v_sig + lr_sig*grad_v_sig';
           
          % ReLU
          g_ReLU = imgs_bias_train(i,:) * w_ReLU;
          y_ReLU = max(g_ReLU,0);
          u_ReLU = y_ReLU*v_ReLU;
          z_ReLU = softmax(u_ReLU');
          grad_w_ReLU =(sign(y_ReLU)'.* (v_ReLU*(t(:,i) - z_ReLU)))* imgs_bias_train(i,:);
          grad_v_ReLU = (t(:,i) - z_ReLU) * y_ReLU;
          w_ReLU = w_ReLU + lr_ReLU*grad_w_ReLU';
          v_ReLU = v_ReLU + lr_ReLU*grad_v_ReLU';
          
          % calculate PoE
          if mod(i,steps) == 0
              % sigmoid
              g_train = imgs_bias_train * w_sig;
              y_train = logsig(g_train);
              u_train = y_train * v_sig;
              z_train = softmax(u_train');
              [~, predi_train_sig] = max(z_train);
              err_train_sig = sum(predi_train_sig - 1 ~= labels_train');
              error_train_sig(h,(e-1)*num_train/steps+floor(i/steps)) = err_train_sig/num_train;
    
              g_test = imgs_bias_test * w_sig;
              y_test = logsig(g_test);
              u_test = y_test * v_sig;
              z_test = softmax(u_test');
              [~, pred_test_sig] = max(z_test);
              err_test_sig = sum(pred_test_sig - 1 ~= labels_test');
              error_test_sig(h,(e-1)*num_train/steps+floor(i/steps)) = err_test_sig/num_test;
            
              % ReLU
              g_train = imgs_bias_train * w_ReLU;
              y_train = max(g_train,0);
              u_train = y_train * v_ReLU;
              z_train = softmax(u_train');
              [~, predi_train_ReLU] = max(z_train);
              err_train_ReLU = sum(predi_train_ReLU - 1 ~= labels_train');
              error_train_ReLU(h,(e-1)*num_train/steps+floor(i/steps)) = err_train_ReLU/num_train;
        
              g_test = imgs_bias_test * w_ReLU;
              y_test = max(g_test,0);
              u_test = y_test * v_ReLU;
              z_test = softmax(u_test');
              [~, pred_test_ReLU] = max(z_test);
              err_test_ReLU = sum(pred_test_ReLU - 1 ~= labels_test');
              error_test_ReLU(h,(e-1)*num_train/steps+floor(i/steps)) = err_test_ReLU/num_test;
          end
      end 
    end
end
%%
figure(1)
for h = 1:3
    subplot(3,2,2*h-1)
    plot(error_train_sig(h,:),LineWidth=1.5); hold on;
    plot(error_test_sig(h,:),LineWidth=1.5,LineStyle='-.'); hold off;
    legend('train set','test set');
    xticks(0:epoch*num_train/steps/5:epoch*num_train/steps);
    xticklabels({'0','5','10','15','20'});
    xlabel('Epoch');
    ylabel('Probability of Error')
    title("Sigmoid with H =  " + H(h));
    grid on;
    disp("The final errors of training set and test set with H = "+H(h)+" are" + ...
        ": "+error_train_sig(h,end)+", "+error_test_sig(h,end));

    subplot(3,2,2*h)
    plot(error_train_ReLU(h,:),LineWidth=1.5); hold on;
    plot(error_test_ReLU(h,:),LineWidth=1.5,LineStyle='-.'); hold off;
    legend('train set','test set');
    xticks(0:epoch*num_train/steps/5:epoch*num_train/steps);
    xticklabels({'0','5','10','15','20'});
    xlabel('Epoch');
    ylabel('Probability of Error')
    title("ReLU with H =  "+ H(h));
    grid on; 

    disp("The final errors of training set and test set with H = "+H(h)+" are" + ...
        ": "+error_train_ReLU(h,end)+", "+error_test_ReLU(h,end));
end