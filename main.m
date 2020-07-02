clear;
close all;
clc;

% load data
train_data_path = './Faces_FA_FB/fa_L';
test_data_path = './Faces_FA_FB/fb_L';
[train_images, train_labels] = ReadFaces(train_data_path);
[test_images, test_labels] = ReadFaces(test_data_path);

train_start_idx = 66;
train_images = train_images(train_start_idx:end);
train_labels = train_labels(train_start_idx:end);

% compute eigen faces
train_data = cell2mat(cellfun(@(img) (reshape(img, [], 1)'), train_images, 'UniformOutput', false)); 
[m, A, eigen_faces, eigen_num] = EigenfaceCore(train_data', 0.95);
eigen_faces_reduced = eigen_faces(:, 1:eigen_num);

% compute projection
train_projection = eigen_faces_reduced' * A;

test_data = cell2mat(cellfun(@(img) (reshape(img, [], 1)'), test_images, 'UniformOutput', false)); 
n_test = size(test_data, 1);
test_projection = eigen_faces_reduced' * (double(test_data') - repmat(m, [1, n_test]));

% do knn
K_array = 1 : 1;
test_accuracy_array = zeros(size(K_array, 2), 1);
for i = 1 : size(K_array, 2)
    K = K_array(i);
    im = test_images{1};
    image_shape = size(im);

    [dist_results, idx_results] = FindKnn(test_projection', train_projection', K);
    results_labels = train_labels(idx_results);

    % compute accuracy
    test_accuracy = ComputeAccuracy(results_labels, test_labels);
    test_accuracy_array(i) = test_accuracy;
end

% compute TP TF and max distance to y=x line
fprintf('min dist: %f    max dist: %f\n', min(dist_results), max(dist_results));
e_thresholds = floor(min(dist_results)):0.001:ceil(max(dist_results));
TP_array = zeros(size(e_thresholds, 2), 1);
FP_array = zeros(size(e_thresholds, 2), 1);
dist_to_line = @(x) (abs(x(1) - x(2)) / sqrt(2));
TP_FP_dist_array = zeros(size(e_thresholds, 2), 1);
for i = 1 : size(e_thresholds, 2)
    [TP, FP] = ComputeTPAndFP(dist_results(:, 1)<e_thresholds(i), results_labels(:, 1)==test_labels);
    TP_array(i) = TP;
    FP_array(i) = FP;
    TP_FP_dist_array(i) = dist_to_line([FP, TP]);
end
match_images = train_images{idx_results};


%% visualize

% show average face
figure('name', 'average face');
imshow(reshape(m, image_shape), []);
title('average face');

% show 10 largest eigenvalues
figure('name', '10 largest eigenvalues');
VisualizeEigenface(eigen_faces(:, 1:10), image_shape);

% show 10 smallest eigenvalues
figure('name', '10 smallest eigenvalues');
VisualizeEigenface(eigen_faces(:, end-9:end), image_shape);

% show CMC
figure('name', 'CMC');
plot(test_accuracy_array);
xlabel('Rank');
ylabel('Performance');

% show faces pair
success_match = results_labels == test_labels;
[~, success_sort_idx] = sort(dist_results(success_match, 1));
success_test_idx = find(success_match);
figure('name','success')
for i = 1 : 3
    test_idx = success_test_idx(success_sort_idx(i));
    subplot(3, 2, 2*i-1);
    imshow(test_images{test_idx})
    title('query image');
    subplot(3, 2, 2*i);
    imshow(train_images{idx_results(test_idx, 1)});
    title('gallery image');
    disp(test_labels(test_idx));
    disp(train_labels(idx_results(test_idx, 1)));
end
fail_match = results_labels ~= test_labels;
fail_test_idx = find(fail_match);
figure('name','fail')
for i = 1 : 3
    test_idx = fail_test_idx(i);
    subplot(3, 2, 2*i-1);
    imshow(test_images{test_idx})
    title('query image');
    subplot(3, 2, 2*i);
    imshow(train_images{idx_results(test_idx, 1)});
    title('gallery image');
    disp(test_labels(test_idx));
    disp(train_labels(idx_results(test_idx, 1)));
end


% show TP and TF curve
figure('name', 'TP FP curve');
hold on;
plot(FP_array, TP_array);
xlabel('false positive rate');
ylabel('true positive rate');
[~, max_idx] = max(TP_FP_dist_array);
plot(FP_array(max_idx), TP_array(max_idx), 'or');
plot(0:0.001:1, 0:0.001:1, 'g--');

% values = spcrv([[FP_array(1) FP_array' FP_array(end)]; [TP_array(1) TP_array' TP_array(end)]], 1);
% plot(values(1,:),values(2,:), 'g');

%% others visualize
pca_ratio_array = [0.8, 0.9, 0.95];
K_array = 1 : 50;
test_accuracy_array = zeros(size(K_array, 2), size(pca_ratio_array, 2));
for pca_idx = 1 : 3
    pca_ratio = pca_ratio_array(pca_idx);
    train_data = cell2mat(cellfun(@(img) (reshape(img, [], 1)'), train_images, 'UniformOutput', false)); 
    [m, A, eigen_faces, eigen_num] = EigenfaceCore(train_data', pca_ratio);
    eigen_faces_reduced = eigen_faces(:, 1:eigen_num);

    % compute projection
    train_projection = eigen_faces_reduced' * A;

    test_data = cell2mat(cellfun(@(img) (reshape(img, [], 1)'), test_images, 'UniformOutput', false)); 
    n_test = size(test_data, 1);
    test_projection = eigen_faces_reduced' * (double(test_data') - repmat(m, [1, n_test]));

    % do knn
    for i = 1 : size(K_array, 2)
        K = K_array(i);
        im = test_images{1};
        image_shape = size(im);

        [dist_results, idx_results] = FindKnn(test_projection', train_projection', K);
        results_labels = train_labels(idx_results);

        % compute accuracy
        test_accuracy = ComputeAccuracy(results_labels, test_labels);
        test_accuracy_array(i, pca_idx) = test_accuracy;
    end
end

% show CMC
figure('name', 'CMC');
color = {'.-r'; 'o-g'; 'x-b'};
hold on;
plot(1:50, test_accuracy_array(:, 1), color{1}, ...
    1:50, test_accuracy_array(:, 2), color{2}, ...
    1:50, test_accuracy_array(:, 3), color{3});
legend(sprintf('%d%%', pca_ratio_array(1) * 100), ...
        sprintf('%d%%', pca_ratio_array(2) * 100), ...
        sprintf('%d%%', pca_ratio_array(3) * 100), 'Location', 'southeast');
xlabel('Rank');
ylabel('Performance');