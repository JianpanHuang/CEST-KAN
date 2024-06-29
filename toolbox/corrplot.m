function [corr_coef] = corrplot(x,y,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,xyrg)
    % Compute correlation coefficient
    corr_coef = corr(x, y);
    
    % Create scatter plot
    % figure, set(gcf,'unit','normalized','Position',[0.2,0.2,0.6,0.6])
    scatter(x, y, mksz,mkcl,mktp);
    hold on;
    
    % Add regression line
    xy = [x;y];
    x_fit = xyrg(1):0.001:xyrg(2);
    p = polyfit(x, y, 1);
    y_fit = polyval(p, x_fit);
    plot(x_fit,x_fit,'.k', x_fit, y_fit, linecl, 'LineWidth',linewth);
    
    % Add correlation coefficient text
    text(xyrg(1)+0.02*(max(xy(:))-min(xy(:))), xyrg(2), ['R = ', num2str(corr_coef)], 'VerticalAlignment', 'top','FontSize',ftsz,'FontWeight','BOLD');
    
    % Add labels and title
    xlabel(xlb);
    ylabel(ylb);
    title(ttl);
    
    % Set aspect ratio to equal
    axis equal;
    
    % Set plot limits
    xlim(xyrg);
    ylim(xyrg);
    
    set(gca,"FontWeight",'BOLD','FontSize',ftsz)
    
    % Turn on grid
    % grid on;
    
    % Hold off
    hold off;
end