%% init
clear

for pt=1:20
  for ps=1:20
  
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


% target
yTarget=[];
for i=1:size(X,1)
for j=1:size(X,2)
    yTarget(i,j)=mean(yTest_mean_cpu(find(idx_valid & xTest(:,1)==0 & xTest(:,2)==0 & xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end

idx_xTrainSource=[];
idx_xTrainTarget=[];
for i=1:25
    for j=1:27
        if ~isnan(ySource(j,i))
            idx_xTrainSource=[idx_xTrainSource;i j];
        end   
        if ~isnan(yTarget(j,i))
            idx_xTrainTarget=[idx_xTrainTarget;i j];
        end   
    end
end


% source
percent=(ps-1)*5/100;
if percent==0, percent=0.01; end
n=length(idx_xTrainSource);
v=randperm(n);
xTrainSource = idx_xTrainSource(v(1:n*percent),:);


% target
percent=(pt-1)*5/100;
if percent==0, percent=0.01; end
n=length(idx_xTrainTarget);
v=randperm(n);
xTrainTarget = idx_xTrainTarget(v(1:n*percent),:);    

%%
T=2;
d=2; % dimension of the conf space
domain = [];
[X1,X2]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]),log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]));
xTest2=[X2(:) X1(:)];
ySourceTest=ySource(:);
yTargetTest=yTarget(:);


idx_source=[];
for i=1:length(xTrainSource)
idx_source(i,1)=find(xTest2(:,1)==X(xTrainSource(i,2),xTrainSource(i,1)) & xTest2(:,2)==Y(xTrainSource(i,2),xTrainSource(i,1)));
end

idx_target=[];
for i=1:length(xTrainTarget)
idx_target(i,1)=find(xTest2(:,1)==X(xTrainTarget(i,2),xTrainTarget(i,1)) & xTest2(:,2)==Y(xTrainTarget(i,2),xTrainTarget(i,1)));
end

observedX=[];
observedX{2}=xTest2(idx_source,:);

observedY=[];
observedY{2}=ySourceTest(idx_source);

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

tic
gps = optmtHyp(gps, x, y);
timetrain=toc;

tic
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);
timetest=toc;

m = reshape(m,[size(xTest2,1) T]);
s2 = reshape(s2,[size(xTest2,1) T]);
s = sqrt(s2);

ms(:,20*(pt-1)+ps)=m(:,1);
ss(:,20*(pt-1)+ps)=s(:,1);
timetrain_mtgp(20*(pt-1)+ps)=timetrain;
timetest_mtgp(20*(pt-1)+ps)=timetest;

yEstimated=reshape(m(:,1),size(X));

end
end

%% prediction error
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

aveErr=reshape(mean(gpAPs1),[20 20]);
contour(aveErr,'ShowText','on')
xticks([1 2:2:20])
xticklabels([1 5:10:95])
yticks([1 2:2:20])
yticklabels([1 5:10:95])
set(0, 'defaultTextInterpreter', 'latex'); 
%hold on; plot(2,4,'rx')
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
fontset

timetrain=reshape(timetrain_mtgp,[20 20]);
timetest=reshape(timetest_mtgp,[20 20]);

figure(2)
contourf(timetrain)
colorbar
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
title('model training time [s]')
xticks([1 2:2:20])
xticklabels([1 5:10:95])
yticks([1 2:2:20])
yticklabels([1 5:10:95])
fontset

figure(3)
contourf(timetest)
colorbar
xticks([1 2:2:20])
xticklabels([1 5:10:95])
yticks([1 2:2:20])
yticklabels([1 5:10:95])
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
title('model evaluation time [s]')
fontset