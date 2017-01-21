colours = {'k', 'r', 'c', 'm', 'b'};
markers = {'o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>', 'p', 'h'};
lines = {'-', ':', '-.', '--'};
psmax=11;
ptmax=10;

pis=10;
pit=1;

ct=12;
cs=0.3;
% for i=1:20
%     for j=1:20
%         if i==1 && j==1
%             effort(i,j)=675*1/100*(ct+cs);
%         else
%             if i==1 && j>1
%                 effort(i,j)=675*1/100*5*(j-1)*ct+675*1/100*cs;
%             else
%                 if j==1 && i>1
%                     effort(i,j)=675*1/100*ct+675*1/100*5*(i-1)*cs;
%                 else
%                     effort(i,j)=675*1/100*5*(j-1)*ct+675*1/100*5*(i-1)*cs;
%                 end
%             end
%         end
%     end
% end
effort=[];
for i=1:psmax
    for j=1:ptmax
effort(i,j)=675*1/100*(ct*j*pit+cs*(i-1)*pis);
    end
end


[A,bb]=prtp([effort(:),aveErr(:)]);

% scatter plot
figure(1)
scatter(effort(:),aveErr(:),'MarkerEdgeColor','b')
for i=1:length(bb)
hold on; plot(effort(bb),aveErr(bb),[colours{1} markers{1}],'MarkerFaceColor','g');
end
xlabel('Measurement cost ($)');
ylabel('Prediction error (mean absolute percentage)');
legend({'samples ($${\mathcal C},pe$$)','Pareto optimal'},'Interpreter','latex');
fontset

idx_pareto=[];
for i=1:length(bb)
    if mod(bb(i),11)>0
   idx_pareto=[idx_pareto;floor(bb(i)/11)+1,mod(bb(i),11)]; 
    else
       idx_pareto=[idx_pareto;floor(bb(i)/11),11];  
    end
end
% indifference diagrams with pareto optimal solutions
figure (2)
contour(aveErr,'ShowText','on')
hold on; h=plot(idx_pareto(:,1),idx_pareto(:,2),'ko');
legend(h,'Pareto optimal');
xticks(1:1:ptmax)
xticklabels(1:1:10)
yticks(1:1:psmax)
yticklabels(0:10:100)
% xticks([1 2:2:20])
% xticklabels([1 5:10:95])
% yticks([1 2:2:20])
% yticklabels([1 5:10:95])
set(0, 'defaultTextInterpreter', 'latex'); 
%hold on; plot(2,4,'rx')
ylabel('$$|{\mathcal D}_s|$$');
xlabel('$$|{\mathcal D}_t|$$');
fontset

% sweet spot solutions
sweet=[];
for i=1:20
    for j=1:20
   if effort(i,j)<=300 && aveErr(i,j)<=10
   sweet=[sweet;i j];
   end
    end
end