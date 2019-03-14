clear all;
close all;
clc;

desc_path = 'PUT YOUR DESCRIPTOR PATH HERE!!!!!!!!!!!';

desc = load(desc_path)

sat_desc = single(desc.sat_global_descriptor);
grd_desc = single(desc.grd_global_descriptor);

% knn search
topk = 200;

sat_desc = sat_desc';
grd_desc = grd_desc';

nb_images = size(sat_desc,2);

recalls = zeros(nb_images, topk);

for i = 1:nb_images

    [ids, dis]= yael_nn(sat_desc, grd_desc(:,i),topk);
    
    % each image the ground truth is the on the diagonal

    df = ids == i;
    
    recalls(i,:) = cumsum(df');
end

mean_recalls = mean(recalls,1);
