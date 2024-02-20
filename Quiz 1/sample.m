%% initializaiton
clc;
close all;
clear;
train = zeros(50*50, 240); test = zeros(50*50, 60);
for i = 0:5
    for j = 1:40
        train(:,i*40+j) = reshape(im2double(imread(strcat('trainset\subset',num2str(i),'\person_',num2str(i+1),'_',num2str(j),'.jpg'))),[],1);
    end
    for o = 1:10
        test(:,i*10+o) = reshape(im2double(imread(strcat('testset\subset',num2str(i+6),'\person_',num2str(i+1),'_',num2str(o),'.jpg'))),[],1);
    end
end
%% PCA
psi = mean(train,2); % average face vector
phi = train - psi; % subtract the mean face
L = phi' * phi; % compute eigenvectors
[V,D] = eig(L);
U = normc(phi*(fliplr(V)));
% plot
figure
for i = 1:16
    subplot(4,4,i);
    imagesc(reshape(U(:,i),50,50));
    colormap(gray(255));
    axis equal
    axis off
end
%% LDA
mu = zeros(2500,6); sigma = zeros(2500,2500,6); 
for i = 1:6
    mu(:,i) = mean(train(:,1+40*(i-1):40+40*(i-1)),2);
    sigma(:,:,i) = cov(train(:,1+40*(i-1):40+40*(i-1))');
end
w=[];
for i =1:5
    for j = i+1:6
        w =[w,(sigma(:,:,i)+sigma(:,:,j)+eye(2500))\(mu(:,i)-mu(:,j))]; % optimal solution
    end
end
% plot
for i = 1:15
    subplot(4,4,i);
    imagesc(reshape(w(:,i),50,50));
    colormap(gray(255));
    axis equal
    axis off
end
%% PoE for PCA,LDA
poe_PCA = PoE(U(:,1:15)'*train, U(:,1:15)'*test);
poe_LDA = PoE(w'*train, w'*test);
%% PoE for PCA+LDA
train_c = U(:,1:30)'*train; test_c = U(:,1:30)'*test;
mu_c = zeros(30,6); sigma_c = zeros(30,30,6);
for i = 1:6
    mu_c(:,i) = mean(train_c(:,1+40*(i-1):40+40*(i-1)),2);
    sigma_c(:,:,i) = cov(train_c(:,1+40*(i-1):40+40*(i-1))');
end
w_c = [];
for i =1:5
    for j = i+1:6
        w_c =[w_c,(sigma_c(:,:,i)+sigma_c(:,:,j)+eye(30))\(mu_c(:,i)-mu_c(:,j))]; % optimal solution
    end
end
poe_All = PoE(w_c'*train_c,w_c'*test_c);
%% Function
function poe = PoE(z,x)
    mu_i = zeros(15,6);
    sigma_i = zeros(15,15,6);
    for i = 1:6
        mu_i(:,i) = mean(z(:,1+40*(i-1):40+40*(i-1)),2);
        sigma_i(:,:,i) = cov(z(:,1+40*(i-1):40+40*(i-1))');
    end
    p = zeros(6,60);
    for i = 1:60
        for j = 1:6
            p(j,i) = (x(:,i)-mu_i(:,j))'*((sigma_i(:,:,j))\(x(:,i)-mu_i(:,j)))+log(det(sigma_i(:,:,j)));
        end
    end
    [~,ind] = min(p);
    number = zeros(1,6);
    for i = 1:60
        if ind(i) ~= ceil(i/10)
            number(ceil(i/10)) = number(ceil(i/10))+1;
        end
    end
    poe = number*10;
end