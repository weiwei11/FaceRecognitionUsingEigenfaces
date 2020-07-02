function [dist_results, idx_results] = FindKnn(query_data, gallery_data, K)
    dist_results = pdist2(query_data, gallery_data, ...
                            'mahalanobis');
    [dist_results , idx_results] = sort(dist_results, 2);
    
    dist_results = dist_results(:, 1:K);
    idx_results = idx_results(:, 1:K);
end
