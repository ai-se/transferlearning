% script for different relationship between source and target
[f, domain, trueMinLoc] = testFunctionFactory('t2');
nMinGridPoints = 1e2;
[xTest, xTestDiff, nTest, nTestPerDim] = makeGrid(domain, nMinGridPoints);

%%g1
g=@(x) awgn(f(x),10,'measured');

figure(1);
h(1)=plot(xTest',f(xTest)','b-');
hold on
h(2)=plot(xTest',g(xTest)','r-');
legend(h,{'$$f(\mathbf{x})$$: target response function','$$g(\mathbf{x})$$: source response function'},'Interpreter','latex')
xlabel('$$x$$');
fontset

%%g2
g=@(x) awgn(f(x),25,'measured')+5000;

figure(2);
h(1)=plot(xTest',f(xTest)','b-');
hold on
h(2)=plot(xTest',g(xTest)','r-');
legend(h,{'$$f(\mathbf{x})$$: target response function','$$g(\mathbf{x})$$: source response function'},'Interpreter','latex')
xlabel('$$x$$');
fontset

%%g3
g=@(x) f_source(x);

figure(3);
h(1)=plot(xTest',f(xTest)','b-');
hold on
for i=1:length(xTest)
    y(i)=g(xTest(i));
end
h(2)=plot(xTest',y','r-');
legend(h,{'$$f(\mathbf{x})$$: target response function','$$g(\mathbf{x})$$: source response function'},'Interpreter','latex')
xlabel('$$x$$');
fontset

%%g4
g=@(x) awgn(5000,10,'measured');

figure(4);
h(1)=plot(xTest',f(xTest)','b-');
hold on
h(2)=plot(xTest',awgn(5000*ones(1,length(xTest)),10,'measured'),'r-');
legend(h,{'$$f(\mathbf{x})$$: target response function','$$g(\mathbf{x})$$: source response function'},'Interpreter','latex')
fontset

function y=f_source(x)
if x<0.5 || x>9.5
    y = awgn(exp(x),15,'measured')+5000;
else
    y = awgn(exp(x),15,'measured');
end
end