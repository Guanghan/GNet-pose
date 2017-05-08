function pcp = computePCP(dist,range)

pcp = zeros(numel(range),size(dist,2)/2+1);
matchAll = zeros(numel(range),size(dist,2)/2*size(dist,3));
for jidx = 1:size(dist,2)/2
    % compute PCP for each threshold
    jidx1 = 2*(jidx-1)+1;
    jidx2 = 2*(jidx-1)+2;
    for k = 1:numel(range)
        match = squeeze(dist(1,jidx1,:))<=range(k) & squeeze(dist(1,jidx2,:)<=range(k));
        pcp(k,jidx) = 100*mean(match);
        matchAll(k,size(dist,3)*(jidx-1)+1:size(dist,3)*(jidx)) = match;
    end
end

% compute average PCP
for k = 1:numel(range)
    pcp(k,end) = 100*mean(matchAll(k,:));
end

end