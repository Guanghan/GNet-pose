% evaluate 
evaluatePCP(7,'OC',false);
evaluatePCP(8,'PC',false);
evaluatePCK(1,'OC',false);
evaluatePCK(6,'PC',false);

% load pre-saved plots
close all;
openfig('./plots/pck-total-lsp-OC.fig');
openfig('./plots/pck-total-lsp-PC.fig');