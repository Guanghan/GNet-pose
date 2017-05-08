function [imgidxs_multi,rectidxs_multi] = getMultiPersonGroups(groups,RELEASE,bTrain)
groups_test = groups(RELEASE.img_train == bTrain);
rectidxs_single = RELEASE.single_person(RELEASE.img_train == bTrain);
imgidxs_multi = [];
rectidxs_multi = cell(0);
for imgidx = 1:length(rectidxs_single)
    rectidxs_groups = groups_test{imgidx};
    rectidxs_single_person = rectidxs_single{imgidx};
    for gidx = 1:length(rectidxs_groups)
        ridxs = rectidxs_groups{gidx};
        if (~ismember(ridxs,rectidxs_single_person))
            imgidxs_multi(end+1) = imgidx;
            rectidxs_multi{end+1} = ridxs;
        end
    end
end
end