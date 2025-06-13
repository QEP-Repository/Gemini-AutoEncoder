function [X, Code, parameters] = gAE_Network(X, VAE, parameters, Predict, Sigma)

    if Predict == 0

        %% Initialization

        % Determine the output size based on the input data
        OutputSize = size(X, 1);

        % Normalize the input data by dividing by the maximum value to scale between -1 and 1
        parameters.norm.weights = dlarray(max(X, [], 'all') - min(X, [], 'all')) / 2;
        X = X ./ parameters.norm.weights;

        % Encoder Section

        % Loop through each layer of the encoder
        for i = 1:length(VAE.Encoder)

            % Initialize weights and biases for the current encoder layer
            parameters.("en" + i).weights = dlarray(normrnd(0, 1, VAE.Encoder(i), size(X, 1))) / 10;
            parameters.("en" + i).bias = dlarray(zeros(VAE.Encoder(i), 1));

            % Apply fully connected layer followed by tanh activation
            X = fullyconnect(X, parameters.("en" + i).weights, parameters.("en" + i).bias);
            X = tanh(X);

        end

        % Code Section

        % Initialize weights and biases for the code layer
        parameters.("code").weights = dlarray(normrnd(0, 1, VAE.Code, size(X, 1)));
        parameters.("code").bias = dlarray(zeros(VAE.Code, 1));

        % Apply fully connected layer followed by tanh activation to get the code
        X = fullyconnect(X, parameters.("code").weights, parameters.("code").bias);
        Code = tanh(X);

        % Add Gaussian noise to the code
        X = normrnd(Code, Sigma);

        % Decoder Section

        % Loop through each layer of the decoder
        for i = 1:length(VAE.Decoder)

            % Initialize weights and biases for the current decoder layer
            parameters.("de" + i).weights = dlarray(normrnd(0, 1, VAE.Decoder(i), size(X, 1))) / 10;
            parameters.("de" + i).bias = dlarray(zeros(VAE.Decoder(i), 1));

            % Apply fully connected layer followed by tanh activation
            X = fullyconnect(X, parameters.("de" + i).weights, parameters.("de" + i).bias);
            X = tanh(X);

        end

        % Output Layer

        % Initialize weights and biases for the output layer
        parameters.("output").weights = dlarray(normrnd(0, 10, OutputSize, size(X, 1)));
        parameters.("output").bias = dlarray(zeros(OutputSize, 1));

        % Apply fully connected layer to get the final output
        X = fullyconnect(X, parameters.("output").weights, parameters.("output").bias);

    elseif Predict == 1 % Autoencoder + Encoder

        %% Initialization

        % Normalize the input data
        X = X ./ parameters.norm.weights;

        % Encoder Section

        % Loop through each layer of the encoder
        for i = 1:length(VAE.Encoder)

            % Apply fully connected layer followed by tanh activation
            X = fullyconnect(X, parameters.("en" + i).weights, parameters.("en" + i).bias);
            X = tanh(X);

        end

        % Code Section

        % Apply fully connected layer to get the code
        X = fullyconnect(X, parameters.("code").weights, parameters.("code").bias);
        Code = X;

        % Add Gaussian noise to the code
        X = normrnd(Code, Sigma);

        % Decoder Section

        % Loop through each layer of the decoder
        for i = 1:length(VAE.Decoder)

            % Apply fully connected layer followed by tanh activation
            X = fullyconnect(X, parameters.("de" + i).weights, parameters.("de" + i).bias);
            X = tanh(X);

        end

        % Apply fully connected layer to get the final output
        X = fullyconnect(X, parameters.("output").weights, parameters.("output").bias);

    elseif Predict == 2 % Decoder (Generative)

        % Loop through each layer of the decoder
        for i = 1:length(VAE.Decoder)

            % Apply fully connected layer followed by tanh activation
            X = fullyconnect(X, parameters.("de" + i).weights, parameters.("de" + i).bias);
            X = tanh(X);

        end

        % Apply fully connected layer to get the final output
        X = fullyconnect(X, parameters.("output").weights, parameters.("output").bias);

    end

end
