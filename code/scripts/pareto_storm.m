%% init
clear

[X1,X2,X3,X4]=ndgrid(log([1,2,3,10,100]),log([1,2,3,6]),log([1,2,3,6]),log([10,100,1000]));
xTest=[X1(:) X2(:) X3(:) X4(:)];
data=csvread('../../experiments/storm/wc-opennebula-c1-12aug.csv');
yTest_latency=data(:,8);
yTest_throughput=data(:,7);

% select objective function
yTest=yTest_latency;

% select source and target
idx_source=find(data(:,1)==1 & data(:,2)==1000);
idx_target=find(data(:,1)==1 & data(:,2)==10000);

T=2;
d=size(xTest,2); % dimension of the conf space

for pt=1:20
  for ps=1:20
  

% source
percent=(ps-1)*5/100;
if percent==0, percent=0.01; end
n=length(idx_source);
v=randperm(n);
idx_train_source = idx_source(v(1:n*percent),:);

% target
percent=(pt-1)*5/100;
if percent==0, percent=0.01; end
n=length(idx_target);
v=randperm(n);
idx_train_target = idx_target(v(1:n*percent),:);    

%%

observedX=[];
idx=mod(idx_train_source,length(xTest));
idx(idx==0)=length(xTest);
observedX{2}=xTest(idx,:);

observedY=[];
observedY{2}=yTest(idx_train_source);

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