function [row,header] = genTablePCP(pcp,name)

assert(length(pcp)==11)
header = sprintf(' Torso & Upper & Lower & Upper & Fore- & Head  & Total %s\n       & Leg   & Leg&  & Arm   & arm   &       &       %s\n','\\','\\');

row = sprintf('%s& %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f & %1.1f %s\n',name,pcp(10),(pcp(2)+pcp(3))/2,(pcp(1)+pcp(4))/2,(pcp(6)+pcp(7))/2,(pcp(5)+pcp(8))/2,pcp(9),pcp(end),'\\');

fprintf('\n%s %s',blanks(length(name)),header);
fprintf('%s\n',row);
end