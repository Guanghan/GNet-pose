function sticks = keypoints2sticks(keypoints)

sticks = zeros(2,20,size(keypoints,3));
sticks(:,1:2,:) = keypoints(:,1:2,:);
sticks(:,3:4,:) = keypoints(:,2:3,:);
sticks(:,5:6,:) = keypoints(:,4:5,:);
sticks(:,7:8,:) = keypoints(:,5:6,:);
sticks(:,9:10,:) = keypoints(:,7:8,:);
sticks(:,11:12,:) = keypoints(:,8:9,:);
sticks(:,13:14,:) = keypoints(:,10:11,:);
sticks(:,15:16,:) = keypoints(:,11:12,:);
sticks(:,17:18,:) = keypoints(:,13:14,:);
sticks(:,19,:) = mean(keypoints(:,3:4,:),2);
sticks(:,20,:) = mean(keypoints(:,9:10,:),2);

end