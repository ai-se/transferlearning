clear
%% init
d=4; % dimension of the conf space
domain = [];
[X1,X2,X3,X4]=ndgrid([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[0:0.05:0.45],[0:0.05:0.45]);
xTest=[X4(:) X3(:) X2(:) X1(:)];
yTest_mean_localization_error=csvread('../../experiments/cobot/mean_localization_error1.csv');
yTest_max_localization_error=csvread('../../experiments/cobot/max_localization_error1.csv');
yTest_mean_cpu=csvread('../../experiments/cobot/mean_cpu_percent1.csv');
yTest_wall_time=csvread('../../experiments/cobot/wall_time1.csv');
yTest=yTest_mean_localization_error(:,2);
xTest=xTest(1:size(yTest,1),:);

% excluding invalid configurations
idx_valid=~(yTest_wall_time(:,2)==100);
xTest=xTest(idx_valid,:);
yTest=yTest(idx_valid,:);

idx_train=(xTest(:,1)==0 & xTest(:,2)==0);
xTrain=xTest(idx_train,:);
yTrain=yTest(idx_train,:);


%xTrainLimit=xTest(40500:end,:);
%yTrainLimit=yTest(40500:end,:);

%training_percent=1;
%numTrain=floor(size(xTest,1)*0.01*training_percent);

%n=size(xTrain,1);
%v=randperm(n);
%idx_train = v(1:numTrain);
%xTrain=xTrainLimit(idx_train,:);
%yTrain=yTrainLimit(idx_train,:);

%% train
%% GP with noisy data
n=size(xTrain,1);
v=randperm(n);
idx_train = v(1:size(xTrain,1)*0.2); % for GP we only use percentage of the data for training
xTrain=xTrain(idx_train,:);
yTrain=yTrain(idx_train,:);
% initialize the prior
gps = covarianceKernelFactory(11, d);
% Bayesian model selection to find hyperparameter
tic
gps = optimizeHyp(gps, xTrain, yTrain);
timetrain=toc;
% Perform GP on the test grid (calculate posterior)
tic
[ym,ys2,m,s2,lp] = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, ...
    gps.likfunc, xTrain, yTrain, xTest, yTest);
timetest=toc;
s = sqrt(s2);
ms(:,1)=m;
ss(:,1)=s;
h(1)=compute_entropy(m,s);

timetrain_gp(:,1)=timetrain;
timetest_gp(:,1)=timetest;

% GP pred after transfer to the overal space
% xTrainLimit=xTest(1:40500,:);
% yTrainLimit=yTest(1:40500,:);
% training_percent=1;
% numTrain=floor(size(xTest,1)*0.01*training_percent);
% v=randperm(n);
% idx_train = v(1:numTrain);
% xTrain=[xTrain; xTrainLimit(idx_train,:)];
% yTrain=[yTrain; yTrainLimit(idx_train,:)];
% % Bayesian model selection to find hyperparameter
% gps = optimizeHyp(gps, xTrain, yTrain);
% % Perform GP on the test grid (calculate posterior)
% [ym ys2 m s2 lp] = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, ...
%     gps.likfunc, xTrain, yTrain, xTest, yTest);
% s = sqrt(s2);
% ms(:,2)=m;
% ss(:,2)=s;
% h(2)=compute_entropy(m,s);

%% MTGP, transfer learning
% initialize the prior
T=3; % number of tasks

gps = covMTKernelFactory(6,T,d);

% observations from other tasks
n=size(xTest,1);
v=randperm(n);
idx_mt_train = v(1:size(idx_train,2));
X=xTest(idx_mt_train,:);
%Y=yTest_wall_time(idx_mt_train,2);
yTransferred=yTest_mean_localization_error(idx_valid,2)+randn*yTest_mean_localization_error(idx_valid,2);
Y=yTransferred(idx_mt_train,:);
observedX{2}=X;
observedY{2}=Y;

v=randperm(n);
idx_mt_train = v(1:size(idx_train,2));
X=xTest(idx_mt_train,:);
yTransferred=yTest_mean_localization_error(idx_valid,2)+rand*yTest_mean_localization_error(idx_valid,2);
Y=yTransferred(idx_mt_train,:);
observedX{3}=X;
observedY{3}=Y;

observedX{1} = xTest(idx_train(1:size(idx_train,2)),:);
observedY{1} = yTest(idx_train(1:size(idx_train,2)),:);

x_test = [xTest ones(size(xTest,1),1)];

x = [observedX{1}, ones(size(observedX{1},1),1)];
y = [observedY{1}];

for i_task = 2:T
    x = [x; observedX{i_task}, ones(size(observedX{i_task},1),1)*i_task];
    y = [y; observedY{i_task}];
    
    x_test=[x_test; xTest ones(size(xTest,1),1)*i_task];
end

% Bayesian model selection to find hyperparameter 
tic
gps = optmtHyp(gps, x, y);
timetrain=toc;

tic
% Perform GP on the test grid (calculate posterior)
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);
timetest=toc;

timetrain_mtgp(:,1)=timetrain;
timetest_mtgp(:,1)=timetest;

% reshape of results
m = reshape(m,[size(xTest,1) T]);
s2 = reshape(s2,[size(xTest,1) T]);
s = sqrt(s2);

ms(:,2)=m(:,1);
ss(:,2)=s(:,1);

%% Random Forest
opts= struct;
opts.depth= 9;
opts.numTrees= 100;
opts.numSplits= 5;
opts.verbose= true;
opts.classifierID= 2; % weak learners to use. Can be an array for mix of weak learners too

tic
mtree= forestTrain(xTrain,yTrain, opts);
timetrain= toc;
tic
yhatTrain(:,1) = forestTest(mtree, xTest);
timetest= toc;

timetrain_rf(:,1)=timetrain;
timetest_rf(:,1)=timetest;

%
opts= struct;
opts.depth= 9;
opts.numTrees= 100;
opts.numSplits= 5;
opts.verbose= true;
opts.classifierID= 1; 

tic
mtree= forestTrain(xTrain,yTrain, opts);
timetrain= toc;
tic
yhatTrain(:,2) = forestTest(mtree, xTest);
timetest= toc;

timetrain_rf(:,2)=timetrain;
timetest_rf(:,2)=timetest;

%
opts= struct;
opts.depth= 12;
opts.numTrees= 100;
opts.numSplits= 5;
opts.verbose= true;
opts.classifierID= [1 2 4]; 

tic
mtree= forestTrain(xTrain,yTrain, opts);
timetrain= toc;
tic
yhatTrain(:,3) = forestTest(mtree, xTest);
timetest= toc;

timetrain_rf(:,3)=timetrain;
timetest_rf(:,3)=timetest;

%% polynomial prediction
datasetSize=size(xTest,1);
actual=yTest;
for n=1:5
    tic;
    reg=MultiPolyRegress(xTrain,yTrain,n);
    timetrain= toc;
    
    tic
    predictedPoly=[];
    for i=1:datasetSize
        NewDataPoint=xTest(i,:);
        NewScores=repmat(NewDataPoint,[length(reg.PowerMatrix) 1]).^reg.PowerMatrix;
        EvalScores=ones(length(reg.PowerMatrix),1);
        for ii=1:size(reg.PowerMatrix,2)
            EvalScores=EvalScores.*NewScores(:,ii);
        end
        predictedPoly=[predictedPoly;reg.Coefficients'*EvalScores]; % The estimate for the new data point.
    end
    timetest= toc;
    
    timetrain_poly(:,n)=timetrain;
    timetest_poly(:,n)=timetest;
    
    polyAE(:,n) = ae(actual, predictedPoly);
    polyAP(:,n)=abs(polyAE(:,n)./actual*100); % absolute percentage
    polyMedianAE(n)=median(polyAE(:,n));
    
    polyMAP(n)=mean(polyAP(:,n));
    polyMedianAP(n)=median(polyAP(:,n));
    
    polyMAE(n) = mae(actual, predictedPoly);
    
    polySE(n)=sum(se(actual,predictedPoly));
    polyTSE(n)=sum(se(actual,mean(actual)));
    polyR2(n)=1-polySE(n)/polyTSE(n);
    
    polyMSE(n) = mse(actual, predictedPoly);
    polyRMSE(n) = rmse(actual, predictedPoly);
    
    polyNMSE(n) = nmse(actual, predictedPoly);
    polyNRMSE(n) = nrmse(actual, predictedPoly);
    
    polyMLSE(n) = msle(actual, predictedPoly);
    polyRMSLE(n) = rmsle(actual, predictedPoly);
    
    polyPerf(n,:)=[polyMAP(n),polyMedianAP(n), polyMAE(n),polyMedianAE(n),polyMSE(n),polyRMSE(n),polyR2(n),polyNMSE(n),polyNRMSE(n)];
    
end

%% acuracy analyses 
% GP
for iter=1:size(ms,2)
    predicted=ms(:,iter);
    
    scoreAE = ae(actual, predicted);
    scoreAP=abs(scoreAE./actual*100); % absolute percentage error
    scoreMedianAE=median(scoreAE);
    
    scoreMAP=mean(scoreAP);
    scoreMedianAP=median(scoreAP);
    
    scoreMAE = mae(actual, predicted);
    
    scoreSE=sum(se(actual,predicted));
    scoreTSE=sum(se(actual,mean(actual)));
    gpR2=1-scoreSE/scoreTSE;
    
    scoreMSE = mse(actual, predicted);
    scoreRMSE = rmse(actual, predicted);
    
    scoreNMSE = nmse(actual, predicted);
    scoreNRMSE = nrmse(actual, predicted);
    
    scoreMLSE = msle(actual, predicted);
    scoreRMSLE = rmsle(actual, predicted);
    
    gpPerfEvol(iter,:)=[scoreMAP,scoreMedianAP, scoreMAE,scoreMedianAE,scoreMSE,scoreRMSE,gpR2,scoreNMSE,scoreNRMSE];
    gpAEs(:,iter)=scoreAE;
    gpAPs(:,iter)=scoreAP;
end

% random forest
for iter=1:size(yhatTrain,2)
    predicted=yhatTrain(:,iter);
    
    scoreAE = ae(actual, predicted);
    scoreAP=abs(scoreAE./actual*100); % absolute percentage error
    scoreMedianAE=median(scoreAE);
    
    scoreMAP=mean(scoreAP);
    scoreMedianAP=median(scoreAP);
    
    scoreMAE = mae(actual, predicted);
    
    scoreSE=sum(se(actual,predicted));
    scoreTSE=sum(se(actual,mean(actual)));
    gpR2=1-scoreSE/scoreTSE;
    
    scoreMSE = mse(actual, predicted);
    scoreRMSE = rmse(actual, predicted);
    
    scoreNMSE = nmse(actual, predicted);
    scoreNRMSE = nrmse(actual, predicted);
    
    scoreMLSE = msle(actual, predicted);
    scoreRMSLE = rmsle(actual, predicted);
    
    rfPerf(iter,:)=[scoreMAP,scoreMedianAP, scoreMAE,scoreMedianAE,scoreMSE,scoreRMSE,gpR2,scoreNMSE,scoreNRMSE];
    rfAEs(:,iter)=scoreAE;
    rfAPs(:,iter)=scoreAP;
end


%% print results
idx=2;
labels={'GP','TL','ply2','ply3','ply4','ply5','RF(9)','RF(IG)','RF(hyb)'};
errorData=[gpAPs(:,1),gpAPs(:,2),polyAP(:,2),polyAP(:,3),polyAP(:,4),polyAP(:,5),rfAPs(:,1),rfAPs(:,2),rfAPs(:,3)];
boxplot(errorData,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('Absolute Percentage Error [%]')
fontset

[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);

%yTest
for i=1:size(X,1)
for j=1:size(X,2)
    avErr(i,j)=mean(yTest(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j))));
end
end
%surf(X,Y,avErr)
contourf(avErr)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset


%GP
t=1;
for i=1:size(X,1)
for j=1:size(X,2)
    avErr(i,j)=mean(gpAPs(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),t));
end
end
%surf(X,Y,avErr)
contourf(avErr)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

%poly
d=5;
for i=1:size(X,1)
for j=1:size(X,2)
    avErr(i,j)=mean(polyAP(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),d));
end
end
%surf(X,Y,avErr)
contourf(avErr)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

%rf
l=3;
for i=1:size(X,1)
for j=1:size(X,2)
    avErr(i,j)=mean(rfAPs(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),l));
end
end
%surf(X,Y,avErr)
contourf(avErr)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

% overhead
timetrain_all=[timetrain_gp,timetrain_mtgp,timetrain_poly(2:end),timetrain_rf];
timetest_all=[timetest_gp,timetest_mtgp,timetest_poly(2:end),timetest_rf];
timetraintest_all=[timetrain_all;timetest_all];
bar(timetraintest_all');
set(gca, 'XTickLabel', labels);
ylabel('Time [s]')
fontset

%%%%%%%%%%%%%%%%%%%
% actual vs predicted sorted
predicted=yhatTrain(:,3);
v=[actual predicted];
[val, I] = sort(v(:,1));
vv=v(I,:);
plot(vv)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
legend('actual','predicted')
ylabel('mean localization error');
xlabel('configuration');
fontset
