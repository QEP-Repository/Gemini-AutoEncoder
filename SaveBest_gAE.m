function [Loss_Best, ind_ep] = SaveBest_gAE(Loss_Best, Loss_Test, Gemini, VAE, ind_ep, parameters, MSE, Loss_test_id, Dkl)
    % This function saves the best model parameters based on the test loss.

    % Retrieve the number of Gemini units
    N = Gemini.Ngemini;

    % Retrieve the buffer window size
    Bu = VAE.Buffer_Window;

    % Retrieve the code size
    CS = VAE.Code;

    % Check if the current test loss is better than the best recorded loss and if the epoch index is greater than or equal to 20
    if (Loss_Test < Loss_Best) && (ind_ep >= 20)

        % Update the best loss with the current test loss
        Loss_Best = Loss_Test;

        % Save the model parameters and related metrics to a .mat file
        save(strcat("NoiseClass_", num2str(Noise_Class), '_', num2str(N), "_", num2str(CS), "_", num2str(Bu), ".mat"), ...
            "parameters", "VAE", "Loss_Test", "MSE", "Loss_test_id", "Dkl");

        % Reset the epoch index
        ind_ep = 0;

        % Display a message indicating that the model parameters have been saved
        disp("Saved");
    end
end
