function pck = computePCK(dist,range)

pck = zeros(numel(range),size(dist,2)+1);

for jidx = 1:size(dist,2)
    % compute PCK for each threshold
    for k = 1:numel(range)
        pck(k,jidx) = 100*mean(squeeze(dist(1,jidx,:))<=range(k));
    end
end

% compute average PCK
for k = 1:numel(range)
    pck(k,end) = 100*mean(reshape(squeeze(dist(1,:,:)),size(dist,2)*size(dist,3),1)<=range(k));
end

end