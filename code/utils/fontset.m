function fontset(fig,size)
if nargin==0
    fig=gcf;
    size=26;
end
set(findall(fig,'-property','MarkerSize'),'MarkerSize',10)
set(findall(fig,'-property','FontSize'),'FontSize',size)
set(findall(fig,'-property','LineWidth'),'LineWidth',2)
set(fig,'Position',[100 200 600 500]);
%legend('Location','Best')
end