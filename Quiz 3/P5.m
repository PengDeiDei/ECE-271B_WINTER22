clear;clc;
% read train and test data
num_train = 20000;
num_test = 10000;
[imgs_train,lbls_train] = readMNIST('MNISTdata/training set/train-images.idx3-ubyte','MNISTdata/training set/train-labels.idx1-ubyte',num_train,0);
[imgs_test,lbls_test] = readMNIST('MNISTdata/test set/t10k-images.idx3-ubyte','MNISTdata/test set/t10k-labels.idx1-ubyte',num_test,0);

%%
labels_train = labelReassign(lbls_train);
labels_test = labelReassign(lbls_test);
classes = size(labels_test,2); % 0-9 classes

dims = size(imgs_train,2); % dimension of image in row vector
thrhs = (0:50)/50; % thresholds
K = 250; % iterations
a_t = cell(K,1); % weak learner

g_train = zeros(num_train,classes);
g_test = zeros(num_test,classes);

margin = zeros(num_train,classes,K); % margin of example x_i -> {y_i*g(x_i)}
index_maxWeights = zeros(classes,K); % index of maximum weights
a_errors = zeros(size(thrhs,2),K);
a_errIndex = zeros(classes,size(thrhs,2),K);
index_thrhs = zeros(classes,K); 

errors_train = zeros(classes,K);
errors_test = zeros(classes,K);
for k = 1: K
    for i = 1:classes
        tic; % star time counter
        % compute the weights
        margin(:,i,k) = labels_train(:,i).*g_train(:,i);
        weights = exp(-1*margin(:,i,k)); % boosting weights
        % count the index of maximum weights for each iteration
        [~,index_maxWeights(i,k)] = max(weights);
        
        % compute negative gradient
        y = repmat(labels_train(:,i),[1,dims]);
        u = cell(size(thrhs,2),1); % decision stump
        for t = 1:size(thrhs,2)
            u{t} = sign(imgs_train-thrhs(t));
            u{t}(u{t} == 0) = 1;
            [a_errors(t,k),a_errIndex(i,t,k)] = min(sum(abs(y-u{t}).*weights));    
        end
        [~,index_thrhs(i,k)] = min(a_errors(:,k));
        a_t{k} = u{index_thrhs(i,k)}(:,a_errIndex(i,index_thrhs(i,k),k));

        % compute step size
        epsilon = sum(weights(labels_train(:,i) ~= a_t{k}))/sum(weights);
        w = 1/2 * log((1 - epsilon) / epsilon);

        % update the learned function
        g_train(:,i) = g_train(:,i) + w*a_t{k};
        
        % compute train error
        h_train = sign(g_train(:,i));
        h_train(h_train == 0) = 1;
        errors_train(i,k) = sum(h_train ~= labels_train(:,i))/num_train;

        % compute test error
        u_test = sign(imgs_test-thrhs(index_thrhs(i,k)));
        u_test(u_test == 0) = 1;
        a_t_test = u_test(:,a_errIndex(i,index_thrhs(i,k),k));
        g_test(:,i) = g_test(:,i) + w*a_t_test;
        h_test = sign(g_test(:,i));
        h_test(h_test == 0) = 1;
        errors_test(i,k) = sum(h_test ~= labels_test(:,i))/num_test;

        toc; % stop time counter
        disp("Iteration #: "+ k +" Classifier #: "+(i-1));
    end
end
%%
[~, finalClasses] = max(g_test.');
final_errors = sum(finalClasses~=(lbls_test+1).')/num_test;
disp("The final error of classifier is: " +final_errors);
%%
figure(1)
for i = 1:classes
    subplot(4,3,i)
    plot(errors_train(i,:));hold on;
    plot(errors_test(i,:));hold off;
    legend('train set','test set');
    xlabel('Iterations');
    ylabel('Probability of Error');
    title("digit "+(i-1));
end

figure(2)
for i = 1:classes
    subplot(4,3,i)
    for k = [5,10,50,100,250]
        cdfplot(margin(:,i,k));hold on;
    end
    hold off;
    legend('5 Iterations','10 Iterations','50 Iterations','100 Iterations','250 Iterations');
    xlabel('margin');
    ylabel('cumulative distribution');
    title("digit "+(i-1));
end

figure(3)
for i = 1:classes
    subplot(5,4,2*i-1)
    plot(index_maxWeights(i,:));
    xlabel('Iterations');
    ylabel('Index of Largest Weight');
    title("digit " + (i-1));

    subplot(5,4,2*i)
    heavist_3 = zeros(28, 3*28);
    temp = tabulate(index_maxWeights(i,:));
    [~, index3] = sort(temp(:, 2),'descend');
    for j = 1:3
        heavist_3(:, j*28-27:j*28) = reshape(imgs_train(temp(index3(j), 1), :), [28, 28])';
    end
    imshow(heavist_3);
end

figure(4)
for i=1:classes
    subplot(4,3,i);
    a = ones(1, 28 * 28) * 128;
    for k = 1:K
        if sum(a_t{k}) > 0
            a(a_errIndex(i, index_thrhs(i, k), k)) = 255;
        else
            a(a_errIndex(i, index_thrhs(i, k), k)) = 0;
        end
    end
    a = reshape(a, [28, 28]);
    imshow(a, [0, 255]);
    title("digit " + (i-1));
end
%%
function re_labels = labelReassign(labels)
    % assign 1 to the images of the specific class label and assign -1 to the rest of classes
    features_num = size(unique(labels),1);
    re_labels = -1*ones(size(labels,1),features_num);
    for i = 1: size(labels,1)
        re_labels(i,labels(i)+1) = 1;
    end
end