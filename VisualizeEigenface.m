function VisualizeEigenface(eigen_faces, image_shape)

    eigen_num = size(eigen_faces, 2);
    plot_rows = ceil(sqrt(eigen_num));
    plot_cols = ceil(eigen_num / plot_rows);
    img = zeros(image_shape);
    for i = 1 : eigen_num
        img(:) = eigen_faces(:, i);
        subplot(plot_rows, plot_cols, i);
        imshow(img, []);
    end
end