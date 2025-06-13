clf % Clear current figure
Config.GPU = 1; % Initialize GPU usage configuration 

% Check if GPU can be used and if it is configured to be used
if canUseGPU && Config.GPU == 1
    gpu = gpuDevice(); % Get GPU device
    reset(gpu) % Reset GPU device

    gpu = gpuDevice(); % Get GPU device again
    disp(gpu) % Display GPU device information
    wait(gpu) % Wait for GPU operations to complete

    clear all; % Clear workspace
    clc; % Clear command window

    Config.GPU = 1; % Set GPU usage configuration to 1
else
    clear all; % Clear workspace
    clc; % Clear command window
    Config.GPU = 0; % Set GPU usage configuration to 0
end

warning off % Turn off warnings

%% Load DataSet
load("Spectra_DataSet.mat") % Load the Spectra dataset

figure(1) % Create a new figure
clf % Clear the current figure

subplot(2,1,1) % Create subplot 1
plot(I_g1_final(:,1), 'lineWidth', 1.5) % Plot the clean data
title("Clean Data") % Set title
xlabel("t [s]") % Set x-axis label
ylabel("y [arb. units]") % Set y-axis label
grid on
grid minor

%% Hyperparameters and SetUp

%%%% Configuration structure : %%%%
% It contains all the operational info of the training process
Config.MaxEpoch = 100; % Maximum number of epochs
Config.IterationsPerEpoch = 500; % Iterations per epoch

%%%% Gemini structure : %%%%
% It contains all the information directly related to Gemini
Gemini.Ngemini = 2; % Number of Gemini (must be > 0). User can change this hyperparameter depending on how many Gemini they want to use for training.

%%%% gAE structure : %%%%
% It contains all the parameters and hyperparameters directly related to the training algorithm
% All of these hyperparameters can be changed
gAE.Encoder = [20 20 20 20 20 20]; % Encoder architecture
gAE.Decoder = flip(gAE.Encoder); % Decoder architecture (symmetric to the encoder)
gAE.Code = 3; % Code size

% Learning hyperparameters
gAE.LearningRate0 = 1e-2; % Initial learning rate
gAE.DecayRate = 1e-3; % Decay rate for learning rate
gAE.MiniBatchSize = 200; % Mini-batch size
gAE.alpha = 1e-4; % Alpha parameter for loss function
gAE.Sigma = 0.05; % Sigma parameter for loss function

% Network and Training Initialization
iteration = 0; % Initialize iteration counter
averageGrad = []; % Initialize average gradient
averageSqGrad = []; % Initialize average squared gradient

%% Model Gradient
accfun = dlaccelerate(@gAE_ModelGradient); % Accelerate the model gradient function

%% Prepare Time Window and data for DL array

dY{1} = dlarray(I_g1_final,'CB');
dY{2} = dlarray(I_g2_final,'CB');

load("Spectra_TestSet.mat")
dY_test = I_test; % this is used in this code only fro plots, while must be used for post-processing analyses

clearvars I_test

%% Test Network and network biases and weights first try values
[~,~,parameters] = gAE_Network(dY{1}(:,1:10), gAE, [], 0, 0); % Initialize network parameters
[dYp,Code] = gAE_Network(dY{1}(:,1:100), gAE, parameters, 1, 0); % Test network with initial parameters

%% Training
% Auxiliary variables for saving the best prediction
ind_ep = 0; % Initialize epoch index
Loss_best = 50; % First saving Loss value

figure() % Create a new figure
clf % Clear the current figure

for epoch = 1 : Config.MaxEpoch % Loop over epochs
    tic
    for i = 1 : Config.IterationsPerEpoch % Loop over iterations per epoch
        % Sample mini-batch
        ind_batch = randsample(size(dY{1},2), gAE.MiniBatchSize); % Randomly sample mini-batch indices

        for i = 1 : Gemini.Ngemini
            dY_T{i} = dY{i}(:,ind_batch); % Time window mini-batch
        end

        % Evaluate gradients
        [gradients,Loss,MSE,Dkl] = dlfeval(accfun, parameters, gAE, dY_T, Gemini, gAE.Sigma); % Evaluate gradients using accelerated function

        % Calculate iterations
        iteration = iteration + 1; % Increment iteration counter

        % Learning rate update
        LearningRate = gAE.LearningRate0./(1+gAE.DecayRate*iteration); % Update learning rate

        % Adam update
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters, gradients, averageGrad, averageSqGrad, iteration, LearningRate); % Update parameters using Adam optimizer
    end

    disp(toc)
    time(epoch) = toc;

    %% Evaluate Predictions
    Loss_true = 0; % Initialize true loss

    [dYp,Code] = gAE_Network(dY_T{1}, gAE, parameters, 1, 0); % Get predictions

    Code = double(extractdata(gather(Code))); % Extract and gather code data

    figure(2) % Create a new figure

    subplot(2,2,1) % Create subplot 2
    plot(epoch, MSE, '.b', 'MarkerSize', 16) % Plot MSE
    title("MSE and \alpha DKL trends along epochs") % Set title
    hold on % Hold on to plot multiple data
    plot(epoch, gAE.alpha.*Dkl, '.r', 'MarkerSize', 16) % Plot alpha DKL
    set(gca,'YScale','log') % Set y-axis scale to log
    legend(["MSE", "\alpha DKL"]) % Set legend

    subplot(2,2,2) % Create subplot 3
    plot3(Code(1,:), Code(2,:), Code(3,:), '.b') % Plot latent code space
    title("Latent Code Space") % Set title

    dYp = gAE_Network(dlarray(dY_test,'CB'), gAE, parameters, 1, 0); % Get predictions for test data
    dYp = double(extractdata(gather(dYp))); % Extract and gather prediction data

    subplot(2,1,2)
    hold off
    plot(dY_test(:,1))
    hold on
    plot(dYp(:,1))
   

    drawnow % Update figure
    
end
