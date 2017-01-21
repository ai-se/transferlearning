colours = {'k', 'r', 'c', 'm', 'b', 'g'};
markers = {'o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>', 'p', 'h'};
lines = {'-', ':', '-.', '--'};
labels={'WC','RS','SOL','cass'};

[mu,sigma,muci,sigmaci] = normfit(aveErr_wc');
e=errorbar(5:10:105,mu,sigma,'-.s');
e.MarkerEdgeColor=colours{2};
e.MarkerFaceColor=colours{2};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
xticks(5:10:105)
xticklabels(0:10:100)
xlim([0 110]);
xlabel('Percentage of training data from source');
ylabel('Mean absolute percentage error on target');

[mu,sigma,muci,sigmaci] = normfit(aveErr_rs');
hold on; e=errorbar(5:10:105,mu,sigma,'--+');
e.MarkerEdgeColor=colours{2};
e.MarkerFaceColor=colours{2};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
fontset

[mu,sigma,muci,sigmaci] = normfit(aveErr_sol');
hold on; e=errorbar(5:10:105,mu,sigma,':x');
e.MarkerEdgeColor=colours{2};
e.MarkerFaceColor=colours{2};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
fontset

[mu,sigma,muci,sigmaci] = normfit(aveErr_cass');
hold on; e=errorbar(5:10:105,mu,sigma,'-d');
e.MarkerEdgeColor=colours{5};
e.MarkerFaceColor=colours{5};
e.MarkerSize = 10;
e.Color = colours{1};
e.CapSize = 10;
fontset
set(gca, 'YScale', 'log')

legend(labels);
fontset