%% init
clear
replication=3;

for p=1:21
    for r=1:replication
    
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

[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
ySource=[];
for i=1:size(X,1)
for j=1:size(X,2)
    ySource(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0.30 & xTest(:,2)==0.45 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end

[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
ySource2=[];
for i=1:size(X,1)
for j=1:size(X,2)
    ySource2(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0.20 & xTest(:,2)==0.35 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end

[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
ySource3=[];
for i=1:size(X,1)
for j=1:size(X,2)
    ySource3(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0.10 & xTest(:,2)==0.25 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end

idx_xTrainSource=[];
idx_xTrainSource2=[];
idx_xTrainSource3=[];

for i=1:25
    for j=1:27
        if ~isnan(ySource(j,i))
            idx_xTrainSource=[idx_xTrainSource;i j];
        end
        if ~isnan(ySource2(j,i))
            idx_xTrainSource2=[idx_xTrainSource2;i j];
        end
        if ~isnan(ySource3(j,i))
            idx_xTrainSource3=[idx_xTrainSource3;i j];
        end
    end
end

% target
yTarget=[];
xTrainTarget=[5,5;5,20;10,10;20,1];
for i=1:size(X,1)
for j=1:size(X,2)
    yTarget(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0 & xTest(:,2)==0 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end

% source 1
percent=(p-1)*5/100;
if percent==0, percent=1; end
n=length(idx_xTrainSource);
v=randperm(n);
xTrainSource = idx_xTrainSource(v(1:n*percent),:);    
xTrainSource=[xTrainSource;xTrainTarget];

% source 2
percent2=(p-1)*5/100;
n=length(idx_xTrainSource2);
v=randperm(n);
xTrainSource2 = idx_xTrainSource2(v(1:n*percent2),:);    
xTrainSource2=[xTrainSource2;xTrainTarget];

% source 3
percent3=(p-1)*5/100;
n=length(idx_xTrainSource3);
v=randperm(n);
xTrainSource3 = idx_xTrainSource3(v(1:n*percent3),:);    
xTrainSource3=[xTrainSource3;xTrainTarget];

%%
T=2;
d=2; % dimension of the conf space
domain = [];
[X1,X2]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
xTest2=[X2(:) X1(:)];
ySourceTest=ySource(:);
ySourceTest2=ySource2(:);
ySourceTest3=ySource3(:);
yTargetTest=yTarget(:);


idx_source=[];
for i=1:length(xTrainSource)
idx_source(i,1)=find(xTest2(:,1)==X(xTrainSource(i,2),xTrainSource(i,1)) & xTest2(:,2)==Y(xTrainSource(i,2),xTrainSource(i,1)));
end

idx_source2=[];
for i=1:length(xTrainSource2)
idx_source2(i,1)=find(xTest2(:,1)==X(xTrainSource2(i,2),xTrainSource2(i,1)) & xTest2(:,2)==Y(xTrainSource2(i,2),xTrainSource2(i,1)));
end

idx_source3=[];
for i=1:length(xTrainSource3)
idx_source3(i,1)=find(xTest2(:,1)==X(xTrainSource3(i,2),xTrainSource3(i,1)) & xTest2(:,2)==Y(xTrainSource3(i,2),xTrainSource3(i,1)));
end

idx_target=[];
for i=1:length(xTrainTarget)
idx_target(i,1)=find(xTest2(:,1)==X(xTrainTarget(i,2),xTrainTarget(i,1)) & xTest2(:,2)==Y(xTrainTarget(i,2),xTrainTarget(i,1)));
end

observedX=[];
observedX{2}=xTest2(idx_source,:);
observedX{3}=xTest2(idx_source2,:);
observedX{4}=xTest2(idx_source3,:);

observedY=[];
observedY{2}=ySourceTest(idx_source);
observedY{3}=ySourceTest2(idx_source2);
observedY{4}=ySourceTest3(idx_source3);

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
ms(:,1)=m(:,1);
ss(:,1)=s(:,1);
[hmt,pmt]=compute_entropy(m(:,1),s(:,1));

yEstimated=reshape(m(:,1),size(X));
pmt=reshape(pmt,size(X));

%% error
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

predic_perf(r,p)=mean(gpAPs);
    end
end
% labels={'GP','TL(T=2)','TL(T=3)'};
% errorData=[gpAPs(:,1),gpAPs(:,2),gpAPs(:,3)];
% figure(5);
% boxplot(errorData,'Labels',labels);
% set(gca,'YTick',[10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4]);
% set(gca, 'YScale', 'log')
% ylabel('Absolute Percentage Error [%]')
% fontset
[mu,sigma,muci,sigmaci] = normfit(predic_perf);
err=muci(2,:)-muci(1,:);
errorbar([1 5:5:100],mu,sigma);
xticks([1 10:10:100])
%set(gca,'XTickLabel',[1 5:5:100]); 
xlabel('Percentage of training data from source');
ylabel('Mean absolute percentage error on target');
fontset