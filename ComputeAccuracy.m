function accuracy = ComputeAccuracy(results_labels, gt_labels)
    K = size(results_labels, 2);
    accuracy_array = any(results_labels == repmat(gt_labels, 1, K), 2);
    accuracy = mean(accuracy_array);
end