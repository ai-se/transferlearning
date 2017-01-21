
%% init
[X1,X2,X3,X4]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),0:0.05:0.45,0:0.05:0.45);
xTest=[X4(:) X3(:) X2(:) X1(:)];
yTest_mean_localization_error=csvread('../../experiments/cobot/mean_localization_error1.csv');
yTest_max_localization_error=csvread('../../experiments/cobot/max_localization_error1.csv');
yTest_mean_cpu=csvread('../../experiments/cobot/mean_cpu_percent1.csv');
yTest_wall_time=csvread('../../experiments/cobot/wall_time1.csv');
yTest=yTest_mean_cpu(:,2);
xTest=xTest(1:size(yTest,1),:);

% excluding invalid configurations
idx_valid=~(yTest_wall_time(:,2)==100);
idx_invalid=(yTest_wall_time(:,2)==100);


% source
xTrainSource=[1,1;1,5;1,10;1,15;1,20;1,25;5,1;5,5;5,10;5,15;5,20;5,25;10,1;10,5;10,10;15,1;15,5;20,1];
[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
ySource=[];
for i=1:size(X,1)
for j=1:size(X,2)
    ySource(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0.30 & xTest(:,2)==0.45 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
figure(1)
contourf(ySource)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainSource)
hold on;
plot(xTrainSource(i,1),xTrainSource(i,2),'kx');
end
fontset

% source 1
xTrainSource1=[5,5;5,20;10,10;20,1];
[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
ySource1=[];
for i=1:size(X,1)
for j=1:size(X,2)
    ySource1(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0.30 & xTest(:,2)==0.45 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
figure(1)
contourf(ySource1)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainSource1)
hold on;
plot(xTrainSource1(i,1),xTrainSource1(i,2),'kx');
end
fontset

% source 2
xTrainSource2=[1,1;1,25;5,1;5,25;10,10;20,1];
%xTrainSource2=[3,1;3,5;3,10;3,15;3,20;3,25;5,1;5,5;5,10;5,15;5,20;10,10;13,1;13,5;20,1;18,5];
[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
ySource2=[];
for i=1:size(X,1)
for j=1:size(X,2)
    ySource2(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0.20 & xTest(:,2)==0.45 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
figure(2)
contourf(ySource2)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainSource2)
hold on;
plot(xTrainSource2(i,1),xTrainSource2(i,2),'kx');
end
fontset


% target
yTarget=[];
xTrainTarget=[5,5;5,20;10,10;20,1];
for i=1:size(X,1)
for j=1:size(X,2)
    yTarget(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0 & xTest(:,2)==0 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
figure(3)
contourf(yTarget)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainTarget)
hold on;
plot(xTrainTarget(i,1),xTrainTarget(i,2),'k*');
end
fontset


%% 2D
T=2;
d=2; % dimension of the conf space
domain = [];
[X1,X2]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
xTest2=[X2(:) X1(:)];
ySourceTest=ySource(:);
ySourceTest2=ySource2(:);
yTargetTest=yTarget(:);


idx_source=[];
for i=1:length(xTrainSource)
idx_source(i,1)=find(xTest2(:,1)==X(xTrainSource(i,2),xTrainSource(i,1)) & xTest2(:,2)==Y(xTrainSource(i,2),xTrainSource(i,1)));
end

idx_source2=[];
for i=1:length(xTrainSource2)
idx_source2(i,1)=find(xTest2(:,1)==X(xTrainSource2(i,2),xTrainSource2(i,1)) & xTest2(:,2)==Y(xTrainSource2(i,2),xTrainSource2(i,1)));
end

idx_target=[];
for i=1:length(xTrainTarget)
idx_target(i,1)=find(xTest2(:,1)==X(xTrainTarget(i,2),xTrainTarget(i,1)) & xTest2(:,2)==Y(xTrainTarget(i,2),xTrainTarget(i,1)));
end

observedX=[];
observedX{2}=xTest2(idx_source,:);
observedX{3}=xTest2(idx_source2,:);

observedY=[];
observedY{2}=ySourceTest(idx_source);
observedY{3}=ySourceTest2(idx_source2);

observedX{1}=xTest2(idx_target,:);
observedY{1}=yTargetTest(idx_target);

x_test = [xTest2 ones(size(xTest2,1),1)];

x = [observedX{1}, ones(size(observedX{1},1),1)];
y = [observedY{1}];

for i_task = 2:T
    x = [x; observedX{i_task}, ones(size(observedX{i_task},1),1)*i_task];
    y = [y; observedY{i_task}];
    
    x_test=[x_test; xTest2 ones(size(xTest2,1),1)*i_task];
end


%% multi-task GP
gps = covMTKernelFactory(5,T,d);
gps = optmtHyp(gps, x, y);
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);
m = reshape(m,[size(xTest2,1) T]);
s2 = reshape(s2,[size(xTest2,1) T]);
s = sqrt(s2);
ms(:,2)=m(:,1);
ss(:,2)=s(:,1);
[hmt,pmt]=compute_entropy(m(:,1),s(:,1));

yEstimated=reshape(m(:,1),size(X));
pmt=reshape(pmt,size(X));

figure(4);
contourf(yEstimated)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainTarget)
hold on;
plot(xTrainTarget(i,1),xTrainTarget(i,2),'k*');
end
fontset

%% 2D
T=3;
d=2; % dimension of the conf space
domain = [];
[X1,X2]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
xTest2=[X2(:) X1(:)];
ySourceTest=ySource(:);
ySourceTest2=ySource2(:);
yTargetTest=yTarget(:);


idx_source=[];
for i=1:length(xTrainSource1)
idx_source(i,1)=find(xTest2(:,1)==X(xTrainSource1(i,2),xTrainSource1(i,1)) & xTest2(:,2)==Y(xTrainSource1(i,2),xTrainSource1(i,1)));
end

idx_source2=[];
for i=1:length(xTrainSource2)
idx_source2(i,1)=find(xTest2(:,1)==X(xTrainSource2(i,2),xTrainSource2(i,1)) & xTest2(:,2)==Y(xTrainSource2(i,2),xTrainSource2(i,1)));
end

idx_target=[];
for i=1:length(xTrainTarget)
idx_target(i,1)=find(xTest2(:,1)==X(xTrainTarget(i,2),xTrainTarget(i,1)) & xTest2(:,2)==Y(xTrainTarget(i,2),xTrainTarget(i,1)));
end

observedX=[];
observedX{2}=xTest2(idx_source,:);
observedX{3}=xTest2(idx_source2,:);

observedY=[];
observedY{2}=ySourceTest(idx_source);
observedY{3}=ySourceTest2(idx_source2);

observedX{1}=xTest2(idx_target,:);
observedY{1}=yTargetTest(idx_target);

x_test = [xTest2 ones(size(xTest2,1),1)];

x = [observedX{1}, ones(size(observedX{1},1),1)];
y = [observedY{1}];

for i_task = 2:T
    x = [x; observedX{i_task}, ones(size(observedX{i_task},1),1)*i_task];
    y = [y; observedY{i_task}];
    
    x_test=[x_test; xTest2 ones(size(xTest2,1),1)*i_task];
end


%% multi-task GP
gps = covMTKernelFactory(5,T,d);
gps = optmtHyp(gps, x, y);
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);
m = reshape(m,[size(xTest2,1) T]);
s2 = reshape(s2,[size(xTest2,1) T]);
s = sqrt(s2);
ms(:,3)=m(:,1);
ss(:,3)=s(:,1);
[hmt,pmt]=compute_entropy(m(:,1),s(:,1));

yEstimated=reshape(m(:,1),size(X));
pmt=reshape(pmt,size(X));

figure(5);
contourf(yEstimated)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainTarget)
hold on;
plot(xTrainTarget(i,1),xTrainTarget(i,2),'k*');
end
fontset


%% GP
xTrain=xTest2(idx_target,:);
yTrain=yTargetTest(idx_target);

gps = covarianceKernelFactory(8, d);
gps = optimizeHyp(gps, xTrain, yTrain);
[ym,ys2,m,s2,lp] = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, ...
    gps.likfunc, xTrain, yTrain, xTest2, yTargetTest);
s = sqrt(s2);
ms(:,1)=m;
ss(:,1)=s;
[hst,pst]=compute_entropy(m,s);

yEstimated=reshape(m,size(X));
pst=reshape(pst,size(X));

figure(6);
contourf(yEstimated)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
zlabel('CPU usage [%]')
for i=1:length(xTrainTarget)
hold on;
plot(xTrainTarget(i,1),xTrainTarget(i,2),'k*');
end
fontset

%% errors

actual=yTargetTest;
gpAEs=[];
gpAPs=[];
for iter=1:size(ms,2)
    predicted=ms(:,iter);
    
    scoreAE = ae(actual(~isnan(actual)), predicted(~isnan(actual)));
    scoreAP=abs(scoreAE./actual(~isnan(actual))*100); % absolute percentage error
    scoreMedianAE=median(scoreAE);
    
    scoreMAP=mean(scoreAP);
    scoreMedianAP=median(scoreAP);
    
    scoreMAE = mae(actual(~isnan(actual)), predicted(~isnan(actual)));
    
    scoreSE=sum(se(actual(~isnan(actual)), predicted(~isnan(actual))));
    scoreTSE=sum(se(actual(~isnan(actual)), predicted(~isnan(actual))));
    gpR2=1-scoreSE/scoreTSE;
    
    scoreMSE = mse(actual(~isnan(actual)), predicted(~isnan(actual)));
    scoreRMSE = rmse(actual(~isnan(actual)), predicted(~isnan(actual)));
    
    scoreNMSE = nmse(actual(~isnan(actual)), predicted(~isnan(actual)));
    scoreNRMSE = nrmse(actual(~isnan(actual)), predicted(~isnan(actual)));
    
    scoreMLSE = msle(actual(~isnan(actual)), predicted(~isnan(actual)));
    scoreRMSLE = rmsle(actual(~isnan(actual)), predicted(~isnan(actual)));
    
    gpPerfEvol(iter,:)=[scoreMAP,scoreMedianAP, scoreMAE,scoreMedianAE,scoreMSE,scoreRMSE,gpR2,scoreNMSE,scoreNRMSE];
    gpAEs(:,iter)=scoreAE;
    gpAPs(:,iter)=scoreAP;
end

labels={'GP','TL(T=2)','TL(T=3)'};
errorData=[gpAPs(:,1),gpAPs(:,2),gpAPs(:,3)];
figure(7);
boxplot(errorData,'Labels',labels);
hold on; plot(mean(errorData),'ko','MarkerFaceColor','k');
set(gca,'YTick',[10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4]);
set(gca, 'YScale', 'log')
ylabel('Absolute Percentage Error [%]')
text(0.56,1000,['mAPE=',sprintf('%.2f',mean(gpAPs(:,1)))]);
text(1.56,1000,['mAPE=',sprintf('%.2f',mean(gpAPs(:,2)))]);
text(2.56,1000,['mAPE=',sprintf('%.2f',mean(gpAPs(:,3)))]);
fontset

%% correlation
idx_nan_source=~isnan(ySourceTest);
idx_nan_target=~isnan(yTargetTest);
idx_corr=idx_nan_source & idx_nan_target;
corrcoef(ySourceTest(idx_corr),ySourceTest(idx_corr))
