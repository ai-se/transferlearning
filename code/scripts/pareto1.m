clear
f=@(x) x.^2+4;
f1=@(x) x./x;
g=@(x) x.^2+19;
nMinGridPoints = 1e2;
domain=[0 14];
%[xTest, xTestDiff, nTest, nTestPerDim] = makeGrid(domain, nMinGridPoints);
xTest=[1:14]';
yTest=g(xTest);
%xs=[1 3 5 9 10 11 12 13 14]';
%xt=[1 3 5]';

actual=yTest;

%% GP with transfer learning
for i=1:14
    for j=1:14
xt=xTest(1:i,:);
xs=xTest(1:j,:);

% preparing data for mtgp
d=1;
T=2;

observedX=[];
observedX{2}=xs;

observedY=[];
observedY{2}=f(xs);

observedX{1}=xt;
observedY{1}=g(xt);

x_test = [xTest ones(size(xTest,1),1)];

x = [observedX{1}, ones(size(observedX{1},1),1)];
y = [observedY{1}];

for i_task = 2:T
    x = [x; observedX{i_task}, ones(size(observedX{i_task},1),1)*i_task];
    y = [y; observedY{i_task}];
    
    x_test=[x_test; xTest ones(size(xTest,1),1)*i_task];
end

gps = covMTKernelFactory(5,T,d);
gps = optmtHyp(gps, x, y);
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);
m = reshape(m,[size(xTest,1) T]);
s2 = reshape(s2,[size(xTest,1) T]);
s = sqrt(s2);
[hmt,pmt]=compute_entropy(m(:,1),s(:,1));

m=m(:,1);
s=s(:,1);

ms(:,14*(i-1)+j)=m;
ss(:,14*(i-1)+j)=s;

    end
end

%% prediction error
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

%% print results
% idx=2;
% labels={'TL'};
% errorData=[gpAPs(:,1)];
% boxplot(errorData,'Labels',labels);
% set(gca, 'YScale', 'log')
% ylabel('Absolute Percentage Error [%]')
% fontset
aveErr=reshape(mean(gpAPs),[14 14]);
contour(aveErr,'ShowText','on')
set(0, 'defaultTextInterpreter', 'latex'); 
hold on; plot(2,4,'rx')
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
fontset