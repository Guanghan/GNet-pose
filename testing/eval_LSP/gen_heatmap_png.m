heatmap = zeros(64, 64)*255;
for i = 1:64
    for j = 1:64
heatmap(i, j) = 1- heatmaps(1, i , j);

    end
end
hmo = HeatMap(heatmap);




colormap('hot')
imagesc(heatmap)
colorbar