clear


design='rr';
pis=10;
pit=1;

psmax=11;
ptmax=1;

replications=3;
cnt=0;

% [X1,X2,X3,X4,X5,X6]=ndgrid([1,3],[1,2,10,100,1000,10000],[1,2,3,10,100],[1,2,3,6],[1,3,6,12],[10,100,1000]);
% xTest=[X1(:) X2(:) X3(:) X4(:) X5(:) X6(:)];
% data=csvread('../../experiments/storm/wc-opennebula-c1-12aug.csv');

% [X1,X2,X3,X4,X5,X6]=ndgrid([1,3],[10,100,1000,10000],[1,2,3,6,9,12,15,18],[1,10,60,120,300],[1e5,1e6,2e6,1e7],[1e3,1e4,1e5]);
% xTest=[X1(:) X2(:) X3(:) X4(:) X5(:) X6(:)];
% data=csvread('../../experiments/storm/rs-opennebula-c2-11aug.csv');

% [X1,X2,X3,X4,X5,X6]=ndgrid([1,3],[1,10,100,1000,10000],[2,3,4,5],[10,100,1000],[10,100,1e3,1e4,1e5,1e6],[1,2,3,6]);
% xTest=[X1(:) X2(:) X3(:) X4(:) X5(:) X6(:)];
% data=csvread('../../experiments/storm/ec2-sol-12aug.csv');
%xTest=xTest(1:length(data),:);

[X1,X2,X3,X4,X5,X6]=ndgrid([0,1],[0,1],[16,24,32,40],[16,24,32,40],[256,384,512,640],[1,3,5,7]);
xTest=[X1(:) X2(:) X3(:) X4(:) X5(:) X6(:)];
data=csvread('../../experiments/cassandra/half-138.csv');


yTest_latency=data(:,8);
yTest_throughput=data(:,7);

% select objective function
yTest=yTest_latency;
yTest_source=yTest+randn^3*yTest.^2+rand^2;

% select source and target
idx_source=[1:1:length(data)]';
idx_target=[1:1:length(data)]';


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

idx=mod(idx_train_target,length(xTest));
idx(idx==0)=length(xTest);
xTrain=xTest(idx,:);
yTrain=yTest(idx_train_target);

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
idx=mod(idx_train_source,length(xTest));
idx(idx==0)=length(xTest);
observedX{2}=xTest(idx,:);

observedY=[];
observedY{2}=yTest_source(idx_train_source);

idx=mod(idx_train_target,length(xTest));
idx(idx==0)=length(xTest);
observedX{1}=xTest(idx,:);
observedY{1}=yTest(idx_train_target);

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
actual=yTest(idx_target);
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

timetrain=reshape(timetrain_model,[psmax ptmax]);
timetest=reshape(timetest_model,[psmax ptmax]);

figure(2)
contourf(timetrain)
colorbar
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
title('model training time [s]')
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
fontset

figure(3)
contourf(timetest)
colorbar
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

figure(4)
contourf(sigmahat)
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