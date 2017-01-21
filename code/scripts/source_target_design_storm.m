clear

design='rr';
pis=10;
pit=1;

psmax=11;
ptmax=10;

replications=1;
cnt=0;

[X1,X2,X3,X4]=ndgrid([1,2,3,10,100],log([1,2,3,6]),log([1,3,6,12]),log([10,100,1000]));
xTest=[X1(:) X2(:) X3(:) X4(:)];
data=csvread('../../experiments/storm/wc-opennebula-c1-12aug.csv');
yTest_latency=data(:,8);
yTest_throughput=data(:,7);

% select objective function
yTest=yTest_latency;

% select source and target
idx_source=find(data(:,1)==1 & data(:,2)==1);
idx_target=find(data(:,1)==1 & data(:,2)==10);
xTest_source=data(idx_source,3:1:6);
xTest_target=data(idx_target,3:1:6);

T=2;
d=size(xTest,2); % dimension of the conf space

for r=1:replications
for pt=10:ptmax
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

%idx=mod(idx_train_target,length(xTest));
%idx(idx==0)=length(xTest);
xTrain=[data(idx_train_target,3),log(data(idx_train_target,4:1:6))];
yTrain=yTest(idx_train_target);

gps = covarianceKernelFactory(12, d);

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
%idx=mod(idx_train_source,length(xTest));
%idx(idx==0)=length(xTest);
observedX{2}=[data(idx_train_source,3),log(data(idx_train_source,4:1:6))];

observedY=[];
observedY{2}=yTest(idx_train_source);

%idx=mod(idx_train_target,length(xTest));
%idx(idx==0)=length(xTest);
observedX{1}=[data(idx_train_target,3),log(data(idx_train_target,4:1:6))];
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
gps = covMTKernelFactory(5,T,d);

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

% print results
aveErr=reshape(mean(gpAPs),[psmax ptmax replications]);
contour(aveErr(:,:,3),'ShowText','on')
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
set(0, 'defaultTextInterpreter', 'latex'); 
%hold on; plot(2,4,'rx')
xlabel('$$|{\mathcal D}_t|$$');
ylabel('$$|{\mathcal D}_s|$$');
fontset

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