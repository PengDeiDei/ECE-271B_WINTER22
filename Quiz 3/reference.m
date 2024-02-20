clear;clc;close all;
TrainNum = 20000; TestNum = 10000; LabelNum = 10;
[TrainImg, TrainLbl] = readMNIST('MNISTdata/training set/train-images.idx3-ubyte','MNISTdata/training set/train-labels.idx1-ubyte', TrainNum, 0);
[TestImg, TestLbl] = readMNIST('MNISTdata/test set/t10k-images.idx3-ubyte','MNISTdata/test set/t10k-labels.idx1-ubyte', TestNum, 0);
target = - ones(TrainNum, LabelNum);
for h = 1:TrainNum
    target(h, TrainLbl(h)+1) = 1;
end
target_test = - ones(TestNum, LabelNum);
for h = 1:TestNum
    target_test(h, TestLbl(h)+1) = 1;
end
dim = size(TrainImg, 2);
iteration = 250;
thres = (0:50) / 50;
g = fix(rand(TrainNum, LabelNum) * 2);
g_test = fix(rand(TestNum, LabelNum) * 2);
omega = zeros(TrainNum, 1);
maxomega = ones(LabelNum, iteration);
margin = zeros(TrainNum, LabelNum, iteration);
learnerDim_error = zeros(size(thres, 2), iteration);
learnerDim_index = zeros(LabelNum, size(thres, 2), iteration);
learnerThres_index = zeros(LabelNum, iteration);
alpha = cell(iteration, 1);
TrainPoe = zeros(LabelNum, iteration);
TestPoe = zeros(LabelNum, iteration);
final = zeros(iteration, 1);
%% adaboost
for T = 1:iteration
    for i = 1:LabelNum
        tic;
        % weight
        margin(:, i, T) = target(:, i) .* g(:, i);
        omega = exp(- margin(:, i, T));
        [~, maxomega(i, T)] = max(omega);
        % weak learner
        y = repmat(target(:, i), [1, dim]);
        u = cell( size(thres, 2), 1);
        for t = 1:51
            u{t} = sign(TrainImg - thres(t));
            u{t}(u{t} == 0) = 1;
            [learnerDim_error(t, T), learnerDim_index(i, t, T)] = min( sum( abs(y - u{t}) .* omega));
        end
        [~, learnerThres_index(i, T)] = min(learnerDim_error(:, T));
        alpha{T} = u{learnerThres_index(i, T)}(:, learnerDim_index(i, learnerThres_index(i, T), T));
        
        % step size
        epsilon = sum( omega(target(:, i) ~= alpha{T})) / sum(omega);
        w = 1/2 * log((1 - epsilon) / epsilon);
        % ensemble learner
        g(:, i) = g(:, i) + w * alpha{T};
        
        % train error
        h = sign( g(:, i));
        h(h == 0) = 1;
        TrainPoe(i, T) = sum(h ~= target(:, i)) / TrainNum;
        % test error
        u_test = sign(TestImg - thres(learnerThres_index(i, T) ));
        u_test(u_test == 0) = 1;
        alpha_test = u_test(:, learnerDim_index(i, learnerThres_index(i, T), T ));
        g_test(:, i) = g_test(:, i) + w * alpha_test;
        h_test = sign( g_test(:, i));
        h_test(h_test == 0) = 1;
        TestPoe(i, T) = sum(h_test ~= target_test(:, i)) / TestNum;
        toc;
        fprintf('current iteration: %d, current classifier: %d   \n', T, i);
    end
    [~, class] = max(transpose(g_test));
    final(T) = sum(class ~= TestLbl' + 1) / TestNum;
end
%% plot
% a
figure;
for i = 1:LabelNum
    subplot(4,3,i);
    plot(TrainPoe(i, :)); hold on
    plot(TestPoe(i, :));
    legend('train', 'test');
    xlabel('iteration');
    ylabel('probability of error');
    title(['digit ', num2str(i-1)]);
end
% b
figure;
for i = 1:LabelNum
    subplot(4,3,i);
    for j = [5, 10, 50, 100, 250]
        cdfplot(margin(:, i, j));
        hold on
    end
    legend('5', '10', '50', '100', '250');
    xlabel('margin');
    ylabel('cdf');
    title(['digit ', num2str(i-1)]);
end
% c
figure;
for i = 1:LabelNum
    subplot(4,3,i);
    plot(maxomega(i,:));
    xlabel('iteration');
    ylabel('index of largest weight');
    title(['digit ', num2str(i-1)]);
end
figure;
for i = 1:LabelNum
    subplot(4,3,i);
    out = zeros(28, 3*28);
    temp = tabulate(maxomega(i,:));
    [~, index3] = sort(temp(:, 2),'descend');
    for j = 1:3
    out(:, j*28-27:j*28) = reshape( TrainImg(temp(index3(j), 1), :), [28, 28])';
    end
    imshow(out)
    title(['digit ', num2str(i-1)]);
end
% d
figure;
for i=1:LabelNum
    subplot(4,3,i);
    a = ones(1, 28 * 28) * 128;
    for T = 1:iteration
        if sum(alpha{T}) > 0
            a(learnerDim_index(i, learnerThres_index(i, T), T)) = 255;
        else
            a(learnerDim_index(i, learnerThres_index(i, T), T)) = 0;
        end
    end
    a = reshape(a, [28, 28]);
    imshow(a, [0, 255]);
    title(['digit ', num2str(i-1)]);
end