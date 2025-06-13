function [gradients, Loss, MSE, Dkl] = gAE_ModelGradient(parameters, VAE, dYbatch, Gemini, Sigma)
    %% MSE Calculation

    % If there is only one Gemini
    if Gemini.Ngemini == 1
        % Get the predicted output and the code from the network
        [dYp, Code] = gAE_Network(dYbatch{1}, VAE, parameters, 1, Sigma);

        % Calculate Mean Squared Error (MSE)
        MSE = mean((dYp - dYbatch{1}).^2, 'all');

        % Calculate mean and standard deviation of the code
        Code_mu = mean(Code, 2);
        Code_STD = std(Code, [], 2);

        % Calculate the KL divergence (Dkl)
        Dkl = log(Code_STD) + (1 + Code_mu.^2) ./ (2 .* Code_STD.^2) - 1/2;
    else
        % Indices for all Gemini
        ind_gemini_total = 1:Gemini.Ngemini;

        % Initialize MSE and Dkl
        MSE = 0;
        Dkl = 0;

        % Loop over each Gemini
        for i = 1 : Gemini.Ngemini
            % Indices for all Gemini except the current one
            ind_gemini = ind_gemini_total;
            ind_gemini(i) = [];

            % Get the predicted output and the code from the network
            [dYp, Code] = gAE_Network(dYbatch{i}, VAE, parameters, 1, Sigma);

            % Calculate mean and standard deviation of the code
            Code_mu = mean(Code, 2);
            Code_STD = std(Code, [], 2);

            % Accumulate KL divergence (Dkl)
            Dkl = Dkl + log(Code_STD) + (1 + Code_mu.^2) ./ (2 .* Code_STD.^2) - 1/2;

            % Loop over the remaining Gemini
            for j = 1 : Gemini.Ngemini - 1
                % Accumulate Mean Squared Error (MSE)
                MSE = MSE + mean((dYp - dYbatch{ind_gemini(j)}).^2, 'all');
            end
        end
    end

    % Calculate the mean of the KL divergence (Dkl)
    Dkl = mean(Dkl);

    %% Gradients Calculation

    % Calculate the total loss
    Loss = MSE + VAE.alpha .* Dkl;

    % Calculate gradients of the loss with respect to the parameters
    gradients = dlgradient(Loss, parameters);
end
