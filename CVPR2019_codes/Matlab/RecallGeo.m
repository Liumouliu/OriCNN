
clear all;
close all;
clc;

desc_path = 'PUT YOUR DESCRIPTOR PATH HERE!!!!!!!!!!!';

desc = load(desc_path)

load('../OriNet_CVACT/CVACT_orientations/ACT_data.mat');


sat_desc = single(desc.sat_global_descriptor);
grd_desc = single(desc.grd_global_descriptor);

% knn search

topk = 200;

sat_desc = sat_desc';
grd_desc = grd_desc';

nb_images = size(sat_desc,2);


recalls_5 = zeros(nb_images, topk);

utms_val = utm(valSetAll.valInd,:);

parfor i = 1:nb_images

    [ids, dis]= yael_nn(sat_desc, grd_desc(:,i),topk);
    
    cur_utm = utms_val(i,:);
    
    retrived_utm = utms_val(ids,:);
    
    L2_dis = cur_utm - retrived_utm;
    
    L2_dis = sqrt(L2_dis(:,1).^2 + L2_dis(:,2).^2);
    
    df = L2_dis <=5;
    recalls_5(i,:) = cumsum(df');

end

mean_recalls_5 = mean(logical(recalls_5),1);
