function [train_data_mean, A, eigen_faces, eigen_num] = EigenfaceCore(train_data, pca_ratio)

    n_data = size(train_data, 2);
    
    % compute mean
    train_data_mean = mean(train_data, 2);
    
    % compute covariance and eigen vectors
    A = double(train_data) - repmat(train_data_mean, [1, n_data]);
    L = A' * A;
    [V, D] = eig(L);
%     standard_deviation = eigen_values .* eigen_values / size(D, 1);
%     variance_proportion = standard_deviation / sum(standard_deviation);

    % sort eigen values
    eigen_values = diag(D);
    [~, eigen_values_idx] = sort(eigen_values, 'descend');
    eigen_values = eigen_values(eigen_values_idx);
    V = V(:, eigen_values_idx);
    
    % compute the number of eigen vectors needs used
    eigen_num = 0;
    cumsum_ratio = cumsum(eigen_values) / sum(eigen_values);
    for i = 1 : numel(cumsum_ratio)
        if cumsum_ratio(i) >= pca_ratio
            eigen_num = i;
            break;
        end
    end
    
    % get eigen vectors
    eigen_vectors = V;

    eigen_faces = A * eigen_vectors;
    
end