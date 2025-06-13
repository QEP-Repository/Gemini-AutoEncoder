function y = ReverseBuffer(x, Window, Res)
    % This function processes the input signal x by reversing it in segments defined by Window and Res.

    % Get the number of columns in the input matrix x
    n = size(x, 2);

    % Create an index array from 1 to n
    i = 1:n;

    % Use the buffer function to segment the index array into a matrix with specified Window and Res
    i = buffer(i, Window, Res);

    % Initialize the output array y with zeros
    y = zeros(1, n);

    % Loop through each element in the index array
    for j = 1:n
        % Find the indices in the segmented matrix i that match the current index j
        indices = i == j;

        % Calculate the mean of elements in x corresponding to these indices and assign it to y(j)
        y(j) = mean(x(indices));
    end
end
