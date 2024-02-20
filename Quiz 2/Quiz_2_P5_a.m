clear;clc;
num_train = 60000;
num_test = 10000;
[imgs_train,labels_train] = readMNIST('training set/train-images.idx3-ubyte','training set/train-labels.idx1-ubyte',num_train,0);
[imgs_test,labels_test] = readMNIST('test set/t10k-images.idx3-ubyte','test set/t10k-labels.idx1-ubyte',num_test,0);

OHElabels_train = OHE(labels_train);

imgs_bias_train = [imgs_train, ones(num_train,1)];
imgs_bias_test = [imgs_test, ones(num_test,1)];
%% initilization
iteration = 3500;
lr = 1e-5; % learning rate
w = randn(size(imgs_bias_train,2),10); % weight
error_train = zeros(iteration,1);
error_test = zeros(iteration,1);

for i  = 1: iteration
    a = imgs_bias_train*w;
    y = softmax(a');
    gradient = (OHElabels_train - y)*imgs_bias_train;
    w = w + lr*gradient';

    [~, predi_train] = max(y);
    err_train = sum(predi_train - 1 ~= labels_train');
    error_train(i) = err_train/num_train;

    a_test = imgs_bias_test*w;
    y_test = softmax(a_test');
    [~, pred_test] = max(y_test);
    err_test = sum(pred_test - 1 ~= labels_test');
    error_test(i) = err_test/num_test;
end
%%
figure(1)
plot(error_train,LineWidth=2); hold on;
plot(error_test,LineWidth=2,LineStyle = '-.'); hold off;
legend('train set','test set');
xlabel('Iteration');
ylabel('Probability of Error');
title('The Probability of Error of the Single Layer NN vs. Itertation');
grid on; axis tight;

disp("The final errors of training set and test set are: " + error_train(iteration) +", " + error_test(iteration));
%%
function OHE_labels = OHE(labels)
    features_num = size(unique(labels),1);
    OHE_labels = zeros(features_num,size(labels,1));
    for i = 1: size(labels,1)
        OHE_labels(labels(i)+1,i) = 1;
    end
end