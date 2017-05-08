function [pck,nseg] = eval_plot_pck(isMatchAll, pidxs, verbose, pslidx)

if (nargin < 2)
    pidxs = 1:14;
end

if (nargin < 3)
    verbose = true;
end

if (nargin < 4)
    pslidx = -1;
end

nMatchAll = 0;
nSegAll = 0;
pck = zeros(1,15);
nseg = zeros(1,15);
idxsAll = [];
for pidx = pidxs
    idxs = find(~isnan(isMatchAll(:,pidx)));
    idxsAll = union(idxsAll,idxs);
    nSeg = sum(~isnan(isMatchAll(:,pidx)));
    nMatch = sum(isMatchAll(:,pidx) == 1);
    nMatchAll = nMatchAll + nMatch;
    nSegAll = nSegAll + nSeg;
    pck(pidx) = nMatch/nSeg*100;
    nseg(pidx) = nSeg;
    if (verbose)
        %fprintf('pidx %d: %f; seg: %d %d\n',pidx,nMatch/nSeg,nMatch,nSeg);
    end
    
end

pck(end) = nMatchAll/nSegAll*100;
nseg(end) = nSegAll;

if pslidx == -1
    if (verbose)
        fprintf('total: %f; seg: %d %d\n',nMatchAll/nSegAll, nMatchAll, nSegAll);
        fprintf(' Head & Shoulder & Elbow & Wrist & Hip & Knee  & Ankle & UBody & Total & # images \\\n');
        if (length(pidxs) == 14)
            fprintf('&%1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f & %1.1f & %1.1f & %1.1f & %d\\\n',(pck(7)+pck(8))/2,(pck(11)+pck(12))/2,(pck(10)+pck(13))/2,(pck(9)+pck(14))/2,(pck(3)+pck(4))/2,(pck(2)+pck(5))/2,(pck(1)+pck(6))/2,mean(pck(9:14)),pck(15),length(idxsAll));
        elseif (length(pidxs) == 6)
            fprintf('& - & %1.1f & %1.1f & %1.1f  & -  & -  & - & %1.1f & %1.1f & %d\\\n',(pck(11)+pck(12))/2,(pck(10)+pck(13))/2,(pck(9)+pck(14))/2, mean(pck(9:14)), pck(15),length(idxsAll));
        end
    end
else
    if (verbose)
%         fprintf('total: %f; seg: %d %d\n',nMatchAll/nSegAll, nMatchAll, nSegAll);
%         fprintf('Poselet & Torso & Upper & Lower & Upper & Fore- & Head  & Total & Total \\\n');
%         fprintf('        &       & Leg   & Leg&  & Arm   & arm   &       & PCP   & images\\\n');
        if (length(pidxs) == 10)
            fprintf('%s & &%1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f & %1.1f & %d\\\n',padZeros(num2str(pslidx),4),pck(5),(pck(2)+pck(3))/2,(pck(1)+pck(4))/2,(pck(8)+pck(9))/2,(pck(7)+pck(10))/2,pck(6),pck(11),length(idxsAll));
        elseif (length(pidxs) == 5)
            fprintf('&%1.1f  & - & - & %1.1f  & %1.1f  & -  & %1.1f & %d\\\n',pck(5),(pck(8)+pck(9))/2,(pck(7)+pck(10))/2, pck(11),length(idxsAll));
        elseif (length(pidxs) == 4)
            fprintf('& - & - & - & %1.1f  & %1.1f  & -  & %1.1f & %d\\\n',(pck(8)+pck(9))/2,(pck(7)+pck(10))/2, pck(11),length(idxsAll));
        end
    end
end

pck = [(pck(7)+pck(8))/2,(pck(11)+pck(12))/2,(pck(10)+pck(13))/2,(pck(9)+pck(14))/2,(pck(3)+pck(4))/2,(pck(2)+pck(5))/2,(pck(1)+pck(6))/2,mean(pck(9:14)),pck(15)];