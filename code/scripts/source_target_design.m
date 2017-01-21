clear

design='rr';
pis=10;
pit=1;

psmax=11;
ptmax=10;

replications=3;

cnt=0;

for r=1:replications
for pt=1:ptmax
  for ps=1:psmax
      cnt=cnt+1;
[X1,X2,X3,X4]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20,log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20,0:0.05:0.45,0:0.05:0.45);
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

xTest=xTest(idx_valid,:);

[X,Y]=meshgrid(log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20,log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20);
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


switch design
    case 'rr'
        % source (random)
        percent=(ps-1)*pis/100;
        if percent==0
            xTrainSource=[];
        else
        n=length(idx_xTrainSource);
        v=randperm(n);
        xTrainSource = idx_xTrainSource(v(1:floor(n*percent)),:);
        end
        
        % target (random)
        percent=(pt)*pit/100;
        if percent==0, percent=0.01; end
        n=length(idx_xTrainTarget);
        v=randperm(n);
        xTrainTarget = idx_xTrainTarget(v(1:floor(n*percent)),:);
        
    case 'lr'
        
        % source (lhs)
        percent=(ps-1)*pis/100;
        if percent==0
            xTrainSource=[];
        else
        n=length(idx_xTrainSource);
        idx_xTrainSource_temp=idx_xTrainSource;
        xn=lhsdesign(floor(n*percent),2);
        lb(1)=1;ub(1)=20;lb(2)=1;ub(2)=27;
        XX = bsxfun(@plus,lb,bsxfun(@times,xn,(ub-lb)));
        for i=1:size(XX,1)
            [M,I]=min(pdist2(XX(i,:),idx_xTrainSource_temp));
            XX(i,:)=idx_xTrainSource_temp(I,:);
            idx_xTrainSource_temp(I,:)=[]; % clear the selected point in order not to select this point again in the initial design
        end
        xTrainSource=XX;
        end
        
        % target (random)
        percent=(pt)*pit/100;
        if percent==0, percent=0.01; end
        n=length(idx_xTrainTarget);
        v=randperm(n);
        xTrainTarget = idx_xTrainTarget(v(1:floor(n*percent)),:);
        
    case 'rl'
        % source (random)
        percent=(ps-1)*pis/100;
        if percent==0
            xTrainSource=[];
        else
        n=length(idx_xTrainSource);
        v=randperm(n);
        xTrainSource = idx_xTrainSource(v(1:floor(n*percent)),:);
        end
        
        % target (lhs)
        percent=(pt)*pit/100;
        if percent==0, percent=0.01; end
        n=length(idx_xTrainTarget);
        idx_xTrainTarget_temp=idx_xTrainTarget;
        xn=lhsdesign(floor(n*percent),2);
        lb(1)=1;ub(1)=20;lb(2)=1;ub(2)=27;
        XX = bsxfun(@plus,lb,bsxfun(@times,xn,(ub-lb)));
        for i=1:size(XX,1)
            [M,I]=min(pdist2(XX(i,:),idx_xTrainTarget_temp));
            XX(i,:)=idx_xTrainTarget_temp(I,:);
            idx_xTrainTarget_temp(I,:)=[]; % clear the selected point in order not to select this point again in the initial design
        end
        xTrainTarget=XX;
        
    case 'll'
        % source (lhs)
        percent=(ps-1)*pis/100;
        if percent==0
            xTrainSource=[];
        else
        n=length(idx_xTrainSource);
        idx_xTrainSource_temp=idx_xTrainSource;
        xn=lhsdesign(floor(n*percent),2);
        lb(1)=1;ub(1)=20;lb(2)=1;ub(2)=27;
        XX = bsxfun(@plus,lb,bsxfun(@times,xn,(ub-lb)));
        for i=1:size(XX,1)
            [M,I]=min(pdist2(XX(i,:),idx_xTrainSource_temp));
            XX(i,:)=idx_xTrainSource_temp(I,:);
            idx_xTrainSource_temp(I,:)=[]; % clear the selected point in order not to select this point again in the initial design
        end
        xTrainSource=XX;
        end
        
        % target (lhs)
        percent=(pt)*pit/100;
        if percent==0, percent=0.01; end
        n=length(idx_xTrainTarget);
        idx_xTrainTarget_temp=idx_xTrainTarget;
        xn=lhsdesign(floor(n*percent),2);
        lb(1)=1;ub(1)=20;lb(2)=1;ub(2)=27;
        XX = bsxfun(@plus,lb,bsxfun(@times,xn,(ub-lb)));
        for i=1:size(XX,1)
            [M,I]=min(pdist2(XX(i,:),idx_xTrainTarget_temp));
            XX(i,:)=idx_xTrainTarget_temp(I,:);
            idx_xTrainTarget_temp(I,:)=[]; % clear the selected point in order not to select this point again in the initial design
        end
        xTrainTarget=XX;
               
end
% adding target samples to the source
% if ~isempty(xTrainSource)
% for i=1:length(xTrainTarget)
% xx=find(idx_xTrainSource(:,1)==xTrainTarget(i,1) & (idx_xTrainSource(:,2)==xTrainTarget(i,2)));
% if ~isempty(xx)
% xTrainSource=[xTrainSource;xTrainTarget(i,:)];
% end
% end
% end

T=2;
d=2; % dimension of the conf space
domain = [];
[X1,X2]=ndgrid(log([1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20,log([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000])/20);
xTest2=[X2(:) X1(:)];
xTest2=xTest2(idx_valid,:);

ySourceTest=ySource(:);
yTargetTest=yTarget(:);

%% GP
if isempty(xTrainSource)
idx_target=[];
for i=1:length(xTrainTarget)
idx_target(i,1)=find(xTest2(:,1)==X(xTrainTarget(i,2),xTrainTarget(i,1)) & xTest2(:,2)==Y(xTrainTarget(i,2),xTrainTarget(i,1)));
end
xTrain=xTest2(idx_target,:);
yTrain=yTargetTest(idx_target);

gps = covarianceKernelFactory(8, d);

tic
gps = optimizeHyp(gps, xTrain, yTrain);
timetrain=toc;

tic
[ym,ys2,m,s2,lp] = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, ...
    gps.likfunc, xTrain, yTrain, xTest2, yTargetTest);
timetest=toc;

s = sqrt(s2);

else
%%
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

% print results
aveErr=reshape(mean(gpAPs),[psmax ptmax replications]);

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

figure(2)
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

figure(3)
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
figure(4)
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