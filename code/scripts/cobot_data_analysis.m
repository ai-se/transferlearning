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
idx_invalid=(yTest_wall_time(:,2)==100);

%boxplot for comparing valid and invalid localization errors
labels={'valid','invalid'};
errorData=[yTest(idx_valid,:)',yTest(idx_invalid,:)'];
grp = [zeros(1,length(idx_valid(idx_valid==1))),ones(1,length(idx_invalid(idx_invalid==1)))];
boxplot(errorData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('max localization error')
fontset

% distributions of response change yTest and in the histfit accordingly
figure
h=histfit(yTest,50,'kernel');
h(1).FaceColor = [.8 .8 1];
h(2).Color = [.2 .2 .2];
xlabel('mean localization error');
ylabel('number of configurations');
fontset

% calculating the variance of measurements
% T = readtable('../../experiments/cobot/measurements.csv','ReadVariableNames',false);
% measurements=[];
% measurements(:,1:3)=cellfun(@(x) str2num(x),T{:,1:3});
% cv=[];
% for i=1:max(measurements(:,1))
%     for j=1:4
%         idx=find(measurements(:,1)==i & measurements(:,2)==j);
%         cv(i,j)=std(measurements(idx,3))/mean(measurements(idx,3));
%     end
% end
cv=csvread('../../experiments/cobot/cv.csv');

%boxplot for Coefficient of variation
labels={'time','cpu', 'max loc. err.', 'mean loc. err.'};
boxplot(cv,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('Coefficient of variation')
fontset


[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);

%yTest
m=4;
avcv=[];
for i=1:size(X,1)
for j=1:size(X,2)
    avcv(i,j)=mean(cv(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),m));
end
end
%surf(X,Y,avErr)
contourf(avcv)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
title('Coefficient of Variation (mean loc. err.)')
fontset

figure
h=histfit(cv(:,4),50,'kernel');
h(1).FaceColor = [.8 .8 1];
h(2).Color = [.2 .2 .2];
xlabel('CV (CPU)');
ylabel('number of configurations');
fontset

%%%%%%%%%%%%%%%%%%%%%%%%%
% where are good configurations
[yTest_sorted,I]=sort(yTest);
idx_good=(I(1:size(yTest,1)*0.01));
idx_bad=(I(end-size(yTest,1)*0.01:end));
xTest_good=xTest(idx_good,:);
xTest_bad=xTest(idx_bad,:);

labels={'good configurations','bad configurations'};
cpuData=[yTest_mean_cpu(idx_good,2)',yTest_mean_cpu(idx_bad,2)'];
grp = [zeros(1,length(idx_good)),ones(1,length(idx_bad))];
boxplot(cpuData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('CPU utilization')
fontset

labels={'good configurations','bad configurations'};
timeData=[yTest_wall_time(idx_good,2)',yTest_wall_time(idx_bad,2)'];
grp = [zeros(1,length(idx_good)),ones(1,length(idx_bad))];
boxplot(timeData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('Localization time')
fontset

labels={'good configurations','bad configurations'};
maxlocData=[yTest_max_localization_error(idx_good,2)',yTest_max_localization_error(idx_bad,2)'];
grp = [zeros(1,length(idx_good)),ones(1,length(idx_bad))];
boxplot(maxlocData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('max localization error')
fontset

labels={'good configurations','bad configurations'};
meanlocData=[yTest_mean_localization_error(idx_good,2)',yTest_mean_localization_error(idx_bad,2)'];
grp = [zeros(1,length(idx_good)),ones(1,length(idx_bad))];
boxplot(meanlocData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('mean localization error')
fontset

labels={'good configurations'};
meanlocData=yTest_mean_localization_error(idx_good,2)';
grp = zeros(1,length(idx_good));
boxplot(meanlocData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('mean localization error')
fontset

labels={'bad configurations'};
meanlocData=yTest_mean_localization_error(idx_bad,2)';
grp = zeros(1,length(idx_bad));
boxplot(meanlocData,grp,'Labels',labels);
set(gca, 'YScale', 'log')
ylabel('mean localization error')
fontset

% bad configurations
[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);
freq=zeros(size(Y,1),size(X,2));
%yTest
for i=1:size(X,1)
for j=1:size(X,2)
    freq(i,j)=size(find(xTest_bad(:,3)==X(i,j) & xTest_bad(:,4)==Y(i,j)),1);
end
end
%surf(X,Y,avErr)
contourf(freq)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

[X,Y]=meshgrid(0:0.05:0.45,0:0.05:0.45);
freq=zeros(size(Y,1),size(X,2));
%yTest
for i=1:size(X,1)
for j=1:size(X,2)
    freq(i,j)=size(find(xTest_bad(:,1)==X(i,j) & xTest_bad(:,2)==Y(i,j)),1);
end
end
%surf(X,Y,avErr)
contourf(freq)
colorbar
xlabel('odometry miscalibration')
ylabel('odometry noise')
fontset


% good configurations
[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);
freq=zeros(size(Y,1),size(X,2));
%yTest
for i=1:size(X,1)
for j=1:size(X,2)
    freq(i,j)=size(find(xTest_good(:,3)==X(i,j) & xTest_good(:,4)==Y(i,j)),1);
end
end
%surf(X,Y,avErr)
contourf(freq)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

[X,Y]=meshgrid(0:0.05:0.45,0:0.05:0.45);
freq=zeros(size(Y,1),size(X,2));
%yTest
for i=1:size(X,1)
for j=1:size(X,2)
    freq(i,j)=size(find(xTest_good(:,1)==X(i,j) & xTest_good(:,2)==Y(i,j)),1);
end
end
%surf(X,Y,avErr)
contourf(freq)
colorbar
xlabel('odometry miscalibration')
ylabel('odometry noise')
fontset

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% effect of configuration parameters on individual responses (cpu)

nParticle=[5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000];
cpu=[];
for i=1:size(nParticle,2)
idx=(xTest(:,3)==nParticle(i));
cpu(i,:)=[nParticle(i), mean(yTest_mean_cpu(idx,2))];
end
plot(cpu(:,1),cpu(:,2));
set(gca, 'XScale', 'log')
xlabel('Number of Particles')
ylabel('CPU utilization')
fontset

nRefinement=[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000];
for i=1:size(nRefinement,2)
idx=(xTest(:,4)==nRefinement(i));
cpu(i,:)=[nRefinement(i),mean(yTest_mean_cpu(idx,2))];
end
plot(cpu(:,1),cpu(:,2));
set(gca, 'XScale', 'log')
xlabel('Number of Refinements')
ylabel('CPU utilization')
fontset

[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);
mean_cpu=[];
for i=1:size(X,1)
for j=1:size(X,2)
    mean_cpu(i,j)=mean(yTest_mean_cpu(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
%surf(X,Y,avErr)
contourf(mean_cpu)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

[X,Y]=meshgrid(unique(xTest(:,1)),unique(xTest(:,2)));
mean_cpu=[];
for i=1:size(X,1)
    for j=1:size(X,2)
    mean_cpu(i,j)=mean(yTest_mean_cpu(find(xTest(:,1)==X(i,j) & xTest(:,2)==Y(i,j)),2));
    end
end
%surf(X,Y,avErr)
contourf(mean_cpu)
colorbar
xlabel('odometry miscalibration')
ylabel('odometry noise')
fontset


[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);
mean_wall_time=[];
for i=1:size(X,1)
for j=1:size(X,2)
    mean_wall_time(i,j)=mean(yTest_wall_time(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
%surf(X,Y,avErr)
contourf(mean_wall_time)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

[X,Y]=meshgrid(unique(xTest(:,1)),unique(xTest(:,2)));
mean_wall_time=[];
for i=1:size(X,1)
    for j=1:size(X,2)
    mean_wall_time(i,j)=mean(yTest_wall_time(find(xTest(:,1)==X(i,j) & xTest(:,2)==Y(i,j)),2));
    end
end
%surf(X,Y,avErr)
contourf(mean_wall_time)
colorbar
xlabel('odometry miscalibration')
ylabel('odometry noise')
fontset

[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);
mean_loc=[];
for i=1:size(X,1)
for j=1:size(X,2)
    mean_loc(i,j)=mean(yTest_mean_localization_error(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
%surf(X,Y,avErr)
contourf(mean_loc)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

[X,Y]=meshgrid(unique(xTest(:,1)),unique(xTest(:,2)));
mean_loc=[];
for i=1:size(X,1)
    for j=1:size(X,2)
    mean_loc(i,j)=mean(yTest_mean_localization_error(find(xTest(:,1)==X(i,j) & xTest(:,2)==Y(i,j)),2));
    end
end
%surf(X,Y,avErr)
contourf(mean_loc)
colorbar
xlabel('odometry miscalibration')
ylabel('odometry noise')
fontset


[X,Y]=meshgrid([5,10,20,30,40,50,70,100,125,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000],[1,3,5,8,10,15,20,35,50,70,100,150,200,250,300,350,400,450,500,750,1000,1500,2000,3500,5000,7500,10000]);
max_loc=[];
for i=1:size(X,1)
for j=1:size(X,2)
    max_loc(i,j)=mean(yTest_max_localization_error(find(xTest(:,3)==X(i,j) & xTest(:,4)==Y(i,j)),2));
end
end
%surf(X,Y,avErr)
contourf(max_loc)
colorbar
xlabel('number of particles')
ylabel('number of refinements')
fontset

[X,Y]=meshgrid(unique(xTest(:,1)),unique(xTest(:,2)));
max_loc=[];
for i=1:size(X,1)
    for j=1:size(X,2)
    max_loc(i,j)=mean(yTest_max_localization_error(find(xTest(:,1)==X(i,j) & xTest(:,2)==Y(i,j)),2));
    end
end
%surf(X,Y,avErr)
contourf(max_loc)
colorbar
xlabel('odometry miscalibration')
ylabel('odometry noise')
fontset

