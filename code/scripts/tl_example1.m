clear
f=@(x) x.^2+4;
f1=@(x) x./x;
g=@(x) x.^2+19;
nMinGridPoints = 1e2;
domain=[0 14];
%[xTest, xTestDiff, nTest, nTestPerDim] = makeGrid(domain, nMinGridPoints);
xTest=[1:14]';
xs=[1 3 5 9 10 11 12 13 14]';
xt=[1 3 5]';


label={'source samples','target samples','$$\hat{f}(\mathbf{x})$$','$$\sigma(\mathbf{x})$$','$$f(\mathbf{x})$$'};
y=f(xs);

figure(1);
h(1)=plot(xs,y,'x');
set(gca,'XTick',xs);
axis([0 14 0 220])
hold on; 
h(2)=plot(xt,g(xt),'+');
set(0, 'defaultTextInterpreter', 'latex'); 
xlabel('$$\mathbf{x}$$');
ylabel('$$f(\mathbf{x})$$');
legend(h,label,'Location','northwest');
fontset

%% GP (without transfer learning)
d=1;
gps = covarianceKernelFactory(11, d);
gps = optimizeHyp(gps, xt, g(xt));
[ym,ys2,m,s2,lp] = gp(gps.hyp, @infExact, gps.meanfunc, gps.covfunc, ...
    gps.likfunc, xt, g(xt), xTest, g(xTest));
s = sqrt(s2);
[hst,pst]=compute_entropy(m,s);

% base figure
figure(2);
h(1)=plot(xs,y,'x');
set(gca,'XTick',xs);
axis([1 14 0 220])
hold on; 
h(2)=plot(xt,g(xt),'+');
set(0, 'defaultTextInterpreter', 'latex'); 
xlabel('$$\mathbf{x}$$');
ylabel('$$f(\mathbf{x})$$');

% observations
fu = [m+2*s; flip(m-2*s,1)];

for kk=2:10
    fu1(:,kk)=[m+2*s/(10-kk+1); flip(m+2*s/(10-kk+2),1)];
    fu1(:,kk+10)=[m-2*s/(10-kk+2); flip(m-2*s/(10-kk+1),1)];
end
fu1(:,1)=[m+2*s/10; flip(m,1)];
fu1(:,11)=[m; flip(m-2*s/10,1)];

for kk=1:10
    fu2(:,kk)=[m+2*kk*s/10; flip(m+2*(kk-1)*s/10,1)];
    fu2(:,kk+10)=[m-2*(kk-1)*s/10; flip(m-2*kk*s/10,1)];
end

for kk=1:10
    hold on; h(4)=fill([xTest; flip(xTest,1)], fu2(:,kk), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk)); % [7 7 7]/8
    h3=fill([xTest; flip(xTest,1)], fu2(:,kk+10), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk));
    %set(gcf,'windowbuttonmotionfcn','Fmotion( ([1 0]*get(gca,''currentp'')*[0;1;0] - min(ylim)) / diff(ylim) )');
    set(h(4),'Linestyle','none')
    set(h3,'Linestyle','none')
end


hold on; h(3)=plot(xTest, m); %plot(obsX, obsY, '*');
h(5)=plot(1:0.1:15,g(1:0.1:15),'c-');
legend(h,label,'Location','northwest','Interpreter','latex');
fontset

%% GP with transfer learning

% base figure
figure(3);
h(1)=plot(xs,y,'x');
set(gca,'XTick',xs);
axis([1 14 0 220])
hold on; 
h(2)=plot(xt,g(xt),'+');
set(0, 'defaultTextInterpreter', 'latex'); 
xlabel('$$\mathbf{x}$$');
ylabel('$$f(\mathbf{x})$$');
%legend(h,label,'Location','northwest');

% preparing data for mtgp
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

% observations
fu = [m+2*s; flip(m-2*s,1)];

for kk=2:10
    fu1(:,kk)=[m+2*s/(10-kk+1); flip(m+2*s/(10-kk+2),1)];
    fu1(:,kk+10)=[m-2*s/(10-kk+2); flip(m-2*s/(10-kk+1),1)];
end
fu1(:,1)=[m+2*s/10; flip(m,1)];
fu1(:,11)=[m; flip(m-2*s/10,1)];

for kk=1:10
    fu2(:,kk)=[m+2*kk*s/10; flip(m+2*(kk-1)*s/10,1)];
    fu2(:,kk+10)=[m-2*(kk-1)*s/10; flip(m-2*kk*s/10,1)];
end

for kk=1:10
    hold on; h2=fill([xTest; flip(xTest,1)], fu2(:,kk), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk)); % [7 7 7]/8
    h3=fill([xTest; flip(xTest,1)], fu2(:,kk+10), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk));
    %set(gcf,'windowbuttonmotionfcn','Fmotion( ([1 0]*get(gca,''currentp'')*[0;1;0] - min(ylim)) / diff(ylim) )');
    set(h2,'Linestyle','none')
    set(h3,'Linestyle','none')
end

hold on; h(3)=plot(xTest, m);
h(5)=plot(1:0.1:15,g(1:0.1:15),'c-');
legend(h,label,'Location','northwest','Interpreter','latex');
fontset

%% GP with transfer learning (negative case)

% base figure
figure(4);
h(1)=plot(xs,f1(xs),'x');
set(gca,'XTick',xs);
axis([1 14 0 220])
hold on; 
h(2)=plot(xt,g(xt),'+');
set(0, 'defaultTextInterpreter', 'latex'); 
xlabel('$$\mathbf{x}$$');
ylabel('$$f(\mathbf{x})$$');
%legend(h,label,'Location','northwest');

% preparing data for mtgp
T=2;

observedX=[];
observedX{2}=xs;

observedY=[];
observedY{2}=f1(xs);

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

% observations
fu = [m+2*s; flip(m-2*s,1)];

for kk=2:10
    fu1(:,kk)=[m+2*s/(10-kk+1); flip(m+2*s/(10-kk+2),1)];
    fu1(:,kk+10)=[m-2*s/(10-kk+2); flip(m-2*s/(10-kk+1),1)];
end
fu1(:,1)=[m+2*s/10; flip(m,1)];
fu1(:,11)=[m; flip(m-2*s/10,1)];

for kk=1:10
    fu2(:,kk)=[m+2*kk*s/10; flip(m+2*(kk-1)*s/10,1)];
    fu2(:,kk+10)=[m-2*(kk-1)*s/10; flip(m-2*kk*s/10,1)];
end

for kk=1:10
    hold on; h2=fill([xTest; flip(xTest,1)], fu2(:,kk), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk)); % [7 7 7]/8
    h3=fill([xTest; flip(xTest,1)], fu2(:,kk+10), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk));
    %set(gcf,'windowbuttonmotionfcn','Fmotion( ([1 0]*get(gca,''currentp'')*[0;1;0] - min(ylim)) / diff(ylim) )');
    set(h2,'Linestyle','none')
    set(h3,'Linestyle','none')
end

hold on; h(3)=plot(xTest, m);
h(5)=plot(1:0.1:15,g(1:0.1:15),'c-');
legend(h,label,'Location','northwest','Interpreter','latex');
fontset