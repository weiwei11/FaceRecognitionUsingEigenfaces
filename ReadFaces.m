function [image_cell, image_labels] = ReadFaces(faces_path)
    flist = dir(fullfile(faces_path,'/*.pgm'));
    image_cell = cell(length(flist), 1);
    image_labels = zeros(length(flist), 1);
    for img_idx = 1 : length(flist)
        img_name = flist(img_idx).name;
        img_path = fullfile(faces_path, img_name);
        
        img = imread(img_path);
        image_cell{img_idx} = img;
        
        % compute image label
        str_array = split(img_name, '_');
        image_labels(img_idx) = str2double(str_array(1));
    end
    fprintf('read faces number: %d\n', length(flist));
end