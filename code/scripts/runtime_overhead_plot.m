colours = {'k', 'r', 'c', 'm', 'b', 'g'};
markers = {'o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>', 'p', 'h'};
lines = {'-', ':', '-.', '--'};
labels={'WC','RS','SOL','cass'};

load('results/seams/train-test/training_testing_wc.mat');
[mu,sigma,muci,sigmaci] = normfit(timetrain');
e=errorbar(5:10:105,mu,sigma,'-.s');
e.MarkerEdgeColor=colours{2};
e.MarkerFaceColor=colours{2};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
xticks(5:10:105)
xticklabels(0:10:100)
xlim([0 110]);
set(0, 'defaultTextInterpreter', 'latex'); 
xlabel('$$|{\mathcal D}_s|$$');
ylabel('Model training time [s]');

load('results/seams/train-test/training_testing_rs.mat');
[mu,sigma,muci,sigmaci] = normfit(timetrain');
hold on; e=errorbar(5:10:105,mu,sigma,'--+');
e.MarkerEdgeColor=colours{2};
e.MarkerFaceColor=colours{2};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
fontset

load('results/seams/train-test/training_testing_sol.mat');
[mu,sigma,muci,sigmaci] = normfit(timetrain');
hold on; e=errorbar(5:10:105,mu,sigma,':x');
e.MarkerEdgeColor=colours{2};
e.MarkerFaceColor=colours{2};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
fontset

load('results/seams/train-test/training_testing_cass.mat');
[mu,sigma,muci,sigmaci] = normfit(timetrain');
hold on; e=errorbar(5:10:105,mu,sigma,'-d');
e.MarkerEdgeColor=colours{5};
e.MarkerFaceColor=colours{5};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
fontset
set(gca, 'YScale', 'log')

legend(labels,'Location','NW');
fontset