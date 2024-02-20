clear;clc;
num_train = 60000;
num_test = 10000;
[imgs_train,labels_train] = readMNIST('training set/train-images.idx3-ubyte','training set/train-labels.idx1-ubyte',num_train,0);
[imgs_test,labels_test] = readMNIST('test set/t10k-images.idx3-ubyte','test set/t10k-labels.idx1-ubyte',num_test,0);

OHElabels_train = OHE(labels_train);

imgs_bias_train = [imgs_train, ones(num_train,1)];
imgs_bias_test = [imgs_test, ones(num_test,1)];
%% (II)
lr = 1e-5;
iteration = 5000;

H = [10,20,50];
error_train = zeros(3,iteration);
error_test = zeros(3,iteration);

for h = 1:3
    w = randn(size(imgs_bias_train,2),H(h));
    v = randn(H(h),10);
   for i = 1:iteration
       g = imgs_bias_train * w;
       y = logsig(g);
       u = y*v;
       z = softmax(u');
       % derivative of logsig y = y.*(1-y)
       grad_w =((y.*(1-y))'.* (v*(OHElabels_train - z)))* imgs_bias_train;
       grad_v = (OHElabels_train - z) * y;
       w = w + lr*grad_w';
       v = v + lr*grad_v';

       [~, predi_train] = max(z);
       err_train = sum(predi_train - 1 ~= labels_train');
       error_train(h,i) = err_train/num_train;

       g_test = imgs_bias_test * w;
       y_test = logsig(g_test);
       u_test = y_test * v;
       z_test = softmax(u_test');
       [~, pred_test] = max(z_test);
       err_test = sum(pred_test - 1 ~= labels_test');
       error_test(h,i) = err_test/num_test;
   end 
end

%% 
figure(1)
for h = 1:3
    subplot(3,1,h)
    plot(error_train(h,:,:),LineWidth=1.5); hold on;
    plot(error_test(h,:,:),LineWidth=1.5,LineStyle='-.'); hold off;
    legend('train set','test set');
    xlabel('Iteration');
    ylabel('Probability of Error')
    title("Sigmoid with H =  "+H(h));
    grid on;
    disp("The final errors of training set and test set with H = "+H(h)+" are" + ...
        ": "+error_train(h,iteration)+", "+error_test(h,iteration));
end
%% (III)
iteration = 3000;
lambda = 0.001;
lr_ReLU = 2e-6;
error_train_ReLU = zeros(3,iteration);
error_test_ReLU = zeros(3,iteration);

for h = 1:3
    w = randn(size(imgs_bias_train,2),H(h));
    v = randn(H(h),10);
   for i = 1:iteration
       g = imgs_bias_train * w;
       y = max(g,0);
       u = y*v;
       z = softmax(u');
       grad_w =(sign(y)'.* (v*(OHElabels_train - z)))* imgs_bias_train;
       grad_v = (OHElabels_train - z) * y;
       w = w + lr_ReLU*grad_w'- 2*lambda*w;
       v = v + lr_ReLU*grad_v'- 2*lambda*v;

       [~, predi_train] = max(z);
       err_train = sum(predi_train - 1 ~= labels_train');
       error_train_ReLU(h,i) = err_train/num_train;

       g_test = imgs_bias_test * w;
       y_test = max(g_test,0);
       u_test = y_test * v;
       z_test = softmax(u_test');
       [~, pred_test] = max(z_test);
       err_test = sum(pred_test - 1 ~= labels_test');
       error_test_ReLU(h,i) = err_test/num_test;
   end 
end
%%
figure(2)
for h = 1:3
    subplot(3,2,2*h-1)
    plot(error_train(h,:,:),LineWidth=1.5); hold on;
    plot(error_test(h,:,:),LineWidth=1.5,LineStyle='-.'); hold off;
    legend('train set','test set');
    xlabel('Iteration');
    ylabel('Probability of Error')
    title("Sigmoid with H =  " + H(h));
    grid on;

    subplot(3,2,2*h)
    plot(error_train_ReLU(h,:,:),LineWidth=1.5); hold on;
    plot(error_test_ReLU(h,:,:),LineWidth=1.5,LineStyle='-.'); hold off;
    legend('train set','test set');
    xlabel('Iteration');
    ylabel('Probability of Error')
    title("ReLU with H =  "+ H(h)+", lambda = 0.001");
    grid on; 

    disp("The final errors of training set and test set with H = "+H(h)+", lambda = 0.001 are" + ...
        ": "+error_train_ReLU(h,iteration)+", "+error_test_ReLU(h,iteration));
end

%% (IV)
lambda = 0.0001;
error_train_ReLU_2 = zeros(3,iteration);
error_test_ReLU_2 = zeros(3,iteration);

for h = 1:3
    w = randn(size(imgs_bias_train,2),H(h));
    v = randn(H(h),10);
   for i = 1:iteration
       g = imgs_bias_train * w;
       y = max(g,0);
       u = y*v;
       z = softmax(u');
       grad_w =(sign(y)'.* (v*(OHElabels_train - z)))* imgs_bias_train;
       grad_v = (OHElabels_train - z) * y;
       w = w + lr_ReLU*grad_w'- 2*lambda*w;
       v = v + lr_ReLU*grad_v'- 2*lambda*v;

       [~, predi_train] = max(z);
       err_train = sum(predi_train - 1 ~= labels_train');
       error_train_ReLU_2(h,i) = err_train/num_train;

       g_test = imgs_bias_test * w;
       y_test = max(g_test,0);
       u_test = y_test * v;
       z_test = softmax(u_test');
       [~, pred_test] = max(z_test);
       err_test = sum(pred_test - 1 ~= labels_test');
       error_test_ReLU_2(h,i) = err_test/num_test;
   end 
end
%%
figure(3)
for h = 1:3
    subplot(3,1,h)
    plot(error_train_ReLU_2(h,:,:),LineWidth=1.5); hold on;
    plot(error_test_ReLU_2(h,:,:),LineWidth=1.5,LineStyle='-.'); hold off;
    legend('train set','test set');
    xlabel('Iteration');
    ylabel('Probability of Error')
    title("ReLU with H =  "+ H(h)+", lambda = 0.0001");
    grid on;

    disp("The final errors of training set and test set with H = "+H(h)+", lambda = 0.0001 are" + ...
        ": "+error_train_ReLU_2(h,iteration)+", "+error_test_ReLU_2(h,iteration));
end
%%
function OHE_labels = OHE(labels)
    features_num = size(unique(labels),1);
    OHE_labels = zeros(features_num,size(labels,1));
    for i = 1: size(labels,1)
        OHE_labels(labels(i)+1,i) = 1;
    end
end