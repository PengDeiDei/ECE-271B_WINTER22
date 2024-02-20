clear;
%% 1.4 (b)
% Condition A: alpha = 10, var = 2
% Condition B: alpha = 2, var = 10
Alpha = [10,2];
Var = [2,10];
Gaussian = [];
figure(1)
for i = 1:2
    subplot(1,2,i)
    mu = [Alpha(i);0];
    Sigma = [1,0;0,Var(i)];
    Gaussian_1 = mvnrnd(mu,Sigma,500);
    Gaussian_2 = mvnrnd(-1*mu,Sigma,500);
    Gaussian = [Gaussian, Gaussian_1, Gaussian_2];

    plot(Gaussian_1(:,1),Gaussian_1(:,2),'o'); hold on;
    plot(Gaussian_2(:,1),Gaussian_2(:,2),'+'); hold off;
    grid on;
    axis tight;
    legend('Class 1','Class 2');
    title("Alpha = " + Alpha(i)+ ", Sigma = " + Var(i));
end
%%
figure(2)
subplot(1,2,1)
[~,~,V_1] = svd(Gaussian(:,1:4));
PCs = Gaussian(:,1:4)* V_1;

plot(Gaussian(:,1),Gaussian(:,2),'o');hold on;
plot(Gaussian(:,3),Gaussian(:,4),'+');
quiver(V_1(1,1),V_1(2,1),10,'black');hold off;
grid on;
axis tight;
legend('Class 1','Class 2');
title("Alpha = " + Alpha(1)+ ", Sigma = " + Var(1));

subplot(1,2,2)
[~,~,V_2] = svd(Gaussian(:,5:8));

plot(Gaussian(:,5),Gaussian(:,6),'o');hold on;
plot(Gaussian(:,7),Gaussian(:,8),'+');
quiver(V_2(1,1),V_2(2,1),10,'black');hold off;
grid on;
axis tight;
legend('Class 1','Class 2');
title("Alpha = " + Alpha(2)+ ", Sigma = " + Var(2));
%%
figure(3)
subplot(1,2,1)
mu = [Alpha(1);0];
Sigma = [1,0;0,Var(1)];
w_1 = inv(Sigma)*(2*mu);
plot(Gaussian(:,1),Gaussian(:,2),'o');hold on;
plot(Gaussian(:,3),Gaussian(:,4),'+');
quiver(1/20*w_1(1),1/20*w_1(2),10,'black'); hold off;
grid on;
axis tight;
legend('Class 1','Class 2');
title("Alpha = " + Alpha(1)+ ", Sigma = " + Var(1));

subplot(1,2,2)
mu = [Alpha(2);0];
Sigma = [1,0;0,Var(2)];
w_2 = inv(Sigma)*(2*mu);
plot(Gaussian(:,5),Gaussian(:,6),'o');hold on;
plot(Gaussian(:,7),Gaussian(:,8),'+');
quiver(1/10*w_2(1),1/10*w_2(2),10,'black'); hold off;
grid on;
axis tight;
legend('Class 1','Class 2');
title("Alpha = " + Alpha(2)+ ", Sigma = " + Var(2));