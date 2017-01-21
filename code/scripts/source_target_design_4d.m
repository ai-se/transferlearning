clear

design='rr';
pis=10;
pit=1;

psmax=11;
ptmax=10;

replications=3;

cnt=0;

[X1,X2,X3,X4]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20,log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20,0:0.05:0.45,0:0.05:0.45);
xTest=[X4(:) X3(:) X2(:) X1(:)];
yTest_mean_localization_error=csvread('../../experiments/cobot/mean_localization_error1.csv');
yTest_max_localization_error=csvread('../../experiments/cobot/max_localization_error1.csv');
yTest_mean_cpu=csvread('../../experiments/cobot/mean_cpu_percent1.csv');
yTest_wall_time=csvread('../../experiments/cobot/wall_time1.csv');
yTest=yTest_mean_cpu(1:2700,2);
xTest=xTest(1:size(yTest,1),:);

% excluding invalid configurations
idx_valid=~(yTest_wall_time(1:2700,2)==100);
idx_invalid=(yTest_wall_time(1:2700,2)==100);

ySource=awgn(yTest(idx_valid),1,5);
yTarget=yTest(idx_valid);
xTest=xTest(idx_valid,:);

% select objective function
yTest_source=ySource;
yTest_target=yTarget;

% select source and target
idx_source=[1:1:length(xTest)]';
idx_target=[1:1:length(xTest)]';

T=2;
d=size(xTest,2); % dimension of the conf space


for r=1:replications
for pt=1:ptmax
  for ps=1:psmax
      cnt=cnt+1;

switch design
    case 'rr'
        % source (random)
        percent=(ps-1)*pis/100;
        if percent==0
            idx_train_source=[];
        else
        n=length(idx_source);
        v=randperm(n);
        idx_train_source = idx_source(v(1:floor(n*percent)),:);
        end
        
        % target (random)
        percent=(pt)*pit/100;
        if percent==0, percent=0.01; end
        n=length(idx_target);
        v=randperm(n);
        idx_train_target = idx_target(v(1:floor(n*percent)),:);           
end
% adding target samples to the source
% if ~isempty(idx_train_source)
% idx_train_source=[idx_train_source;idx_train_target];
% end

%% GP
if isempty(idx_train_source)

xTrain=xTest(idx_train_target,:);
yTrain=yTest_target(idx_train_target);

gps = covarianceKernelFactory(11, d);

tic
gps = optimizeHyp(gps, xTrain, yTrain);
timetrain=toc;

tic
[m,s2] = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, ...
    gps.likfunc, xTrain, yTrain, xTest);
timetest=toc;
s = sqrt(s2);

else
%%
observedX=[];
observedX{2}=xTest(idx_train_source,:);

observedY=[];
observedY{2}=yTest_source(idx_train_source);

observedX{1}=xTest(idx_train_target,:);
observedY{1}=yTest_target(idx_train_target);

x_test = [xTest ones(size(xTest,1),1)];

x = [observedX{1}, ones(size(observedX{1},1),1)];
y = [observedY{1}];

for i_task = 2:T
    x = [x; observedX{i_task}, ones(size(observedX{i_task},1),1)*i_task];
    y = [y; observedY{i_task}];
    
    x_test=[x_test; xTest ones(size(xTest,1),1)*i_task];
end

%% multi-task GP
gps = covMTKernelFactory(6,T,d);

tic
gps = optmtHyp(gps, x, y);
timetrain=toc;

tic
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);
timetest=toc;

m = reshape(m,[size(xTest,1) T]);
s2 = reshape(s2,[size(xTest,1) T]);
s = sqrt(s2);
end


%% save estimations and train-test time
ms(:,cnt)=m(:,1);
ss(:,cnt)=s(:,1);
timetrain_model(cnt)=timetrain;
timetest_model(cnt)=timetest;

end
end
end

%% prediction error
actual=yTest_target(idx_target);
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

% print results
aveErr=reshape(mean(gpAPs),[psmax ptmax replications]);

figure(1)
[C,h]=contour(mean(aveErr,3),'ShowText','on');
clabel(C,h,'FontSize',22,'FontSmoothing','on')
title('$$\mu(pe)$$')
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
set(0, 'defaultTextInterpreter', 'latex'); 
%hold on; plot(2,4,'rx')
xlabel('$$|{\mathcal D}_t|$$');
ylabel('$$|{\mathcal D}_s|$$');
fontset

figure(2)
[C,h]=contourf(std(aveErr,0,3),'ShowText','on');
clabel(C,h,'FontSize',22)
title('$$\sigma(pe)$$')
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
set(0, 'defaultTextInterpreter', 'latex'); 
%hold on; plot(2,4,'rx')
xlabel('$$|{\mathcal D}_t|$$');
ylabel('$$|{\mathcal D}_s|$$');
fontset


timetrain=reshape(timetrain_model,[psmax ptmax replications]);
timetest=reshape(timetest_model,[psmax ptmax replications]);

figure(3)
[C,h]=contourf(mean(timetrain,3),'ShowText','on');
clabel(C,h,'FontSize',22)
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
title('model training time [s]')
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
fontset

figure(4)
[C,h]=contourf(mean(timetest,3),'ShowText','on','ShowText','on');
clabel(C,h,'FontSize',22)
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
title('model evaluation time [s]')
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
fontset


for pt=1:ptmax
  for ps=1:psmax
      [muhat(ps,pt),sigmahat(ps,pt)]=normfit(squeeze(aveErr(ps,pt,:)));
  end
end
figure(5)
contourf(sigmahat)
title('$$\sigma(pe)$$')
colorbar
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
set(0, 'defaultTextInterpreter', 'latex'); 
%hold on; plot(2,4,'rx')
xlabel('$$|{\mathcal D}_t|$$');
ylabel('$$|{\mathcal D}_s|$$');
fontset