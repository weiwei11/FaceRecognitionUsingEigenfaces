function [TP, FP] = ComputeTPAndFP(predict_results, gt_results)
    TP = sum(predict_results(gt_results==1) == 1) / sum(gt_results == 1);
    FP = sum(predict_results(gt_results==0) == 1) / sum(gt_results == 0);
end