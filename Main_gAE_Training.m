clf % Clear current figure
Config.GPU = 0; % Initialize GPU usage configuration to 0

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

%% Load Clean DataSet
load("Lorenz_0_DataSet.mat") % Load the Lorenz dataset

figure(1) % Create a new figure
clf % Clear the current figure

subplot(2,1,1) % Create subplot 1
plot(Lorenz.t, Lorenz.y, 'k', 'lineWidth', 1.5) % Plot the clean data
title("Clean Data") % Set title
xlabel("t [s]") % Set x-axis label
ylabel("y [arb. units]") % Set y-axis label

%% Hyperparameters and SetUp

%%%% Configuration structure : %%%%
% It contains all the operational info of the training process
Config.MaxEpoch = 10; % Maximum number of epochs
Config.IterationsPerEpoch = 100; % Iterations per epoch

%%%% Gemini structure : %%%%
% It contains all the information directly related to Gemini
Gemini.Ngemini = 10; % Number of Gemini (must be > 0). User can change this hyperparameter depending on how many Gemini they want to use for training.

%%%% gAE structure : %%%%
% It contains all the parameters and hyperparameters directly related to the training algorithm
% All of these hyperparameters can be changed
gAE.Buffer_Window = 100; % Buffer window size
gAE.Encoder = [20 20 20 20 20 20]; % Encoder architecture
gAE.Decoder = flip(gAE.Encoder); % Decoder architecture (symmetric to the encoder)
gAE.Code = 3; % Code size

% Learning hyperparameters
gAE.LearningRate0 = 1e-2; % Initial learning rate
gAE.DecayRate = 1e-3; % Decay rate for learning rate
gAE.MiniBatchSize = 500; % Mini-batch size
gAE.alpha = 1e-4; % Alpha parameter for loss function
gAE.Sigma = 0.05; % Sigma parameter for loss function

% Network and Training Initialization
iteration = 0; % Initialize iteration counter
averageGrad = []; % Initialize average gradient
averageSqGrad = []; % Initialize average squared gradient

%% Signal pre-processing and noise addition

%%% Gemini Creation %%%
% Derivation of hyperparameters for noise addition
dt = 1e-2; % Time step
t = min(Lorenz.t):dt:max(Lorenz.t); % Time variable vector is reduced
y0 = interp1(Lorenz.t, Lorenz.y, t); % Y variable vector is consequently reduced
Gemini.Length = size(y0, 2); % Length of the signal

for i = 1 : Gemini.Ngemini
    s = (movmean(square(5*t+normrnd(0,2/dt),10),10)+1)/2; % Create noise signal
    s = sin(20*2*pi*t+normrnd(0,pi)).*s; % Modulate noise signal
    yn = y0 + 20*s; % Add noise to the original signal
    y{i} = normrnd(yn,1); % Entire signals with added noise
    clear yn % Clear temporary variable
end

subplot(2,1,2) % Create subplot 2
for i = 1 : Gemini.Ngemini
    plot(t, y{i}) % Plot noisy signals
    hold on % Hold on to plot multiple signals
end
plot(Lorenz.t, Lorenz.y, 'k','lineWidth', 1.5) % Plot original clean signal
title("Noise-added Gemini") % Set title
xlabel("t [s]") % Set x-axis label
ylabel("y [arb. units]") % Set y-axis label
legend(["y_1","y_2","y_3","y_4","y_5","y_0"]) % Set legend

%%% Test signals creation %%%
% These signals are used during training to both evaluate the metrics of interest (not used for training, but only for evaluation) and to plot predicted vs reference signals

t_test = t; % Test time vector
y0_test = y0; % Test clean signal

s = (movmean(square(5*t+normrnd(0,2/dt),10),10)+1)/2; % Create noise signal
s = sin(20*2*pi*t+normrnd(0,pi)).*s; % Modulate noise signal

yn_test = y0_test + 20*s; % Add noise to the test clean signal

T = buffer(yn_test, gAE.Buffer_Window, gAE.Buffer_Window-1); % Create buffer for test noisy signal

dY_test = dlarray(T,'CB'); % Convert to dlarray for deep learning

clearvars t s dt T % Clear temporary variables

%% Model Gradient
accfun = dlaccelerate(@gAE_ModelGradient); % Accelerate the model gradient function

%% Prepare Time Window and data for DL array
for i = 1 : Gemini.Ngemini
    T = buffer(y{i}, gAE.Buffer_Window, gAE.Buffer_Window-1); % Create buffer for each noisy signal

    Y{i} = T; % Time window signals

    % Deep learning array
    dY{i} = dlarray(T,'CB'); % Convert to dlarray for deep learning
end

clear T % Clear temporary variable

%% Test Network and network biases and weights first try values
[~,~,parameters] = gAE_Network(dY{1}, gAE, [], 0, 0); % Initialize network parameters
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
        ind_batch = randsample(Gemini.Length, gAE.MiniBatchSize); % Randomly sample mini-batch indices

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

    subplot(3,3,2) % Create subplot 2
    plot(epoch, MSE, '.b', 'MarkerSize', 16) % Plot MSE
    title("MSE and \alpha DKL trends along epochs") % Set title
    hold on % Hold on to plot multiple data
    plot(epoch, gAE.alpha.*Dkl, '.r', 'MarkerSize', 16) % Plot alpha DKL
    set(gca,'YScale','log') % Set y-axis scale to log
    legend(["MSE", "\alpha DKL"]) % Set legend

    subplot(3,3,3) % Create subplot 3
    plot3(Code(1,:), Code(2,:), Code(3,:), '.b') % Plot latent code space
    title("Latent Code Space") % Set title

    dYp = gAE_Network(dlarray(dY_test,'CB'), gAE, parameters, 1, 0); % Get predictions for test data
    dYp = double(extractdata(gather(dYp))); % Extract and gather prediction data
    dYp = ReverseBuffer(dYp, gAE.Buffer_Window, gAE.Buffer_Window-1); % Reverse buffer to get entire signal

    %% Metrics evaluation
    Loss_test_real = mean((dYp-yn_test).^2,'all'); % Calculate real test loss
    Loss_test_ideal = mean((dYp-y0_test).^2,'all'); % Calculate ideal test loss

    subplot(3,1,2) % Create subplot 2
    hold off % Hold off to clear previous plots
    plot(t_test, yn_test) % Plot noisy test signal
    title("Noisy test signal vs Prediction") % Set title
    hold on % Hold on to plot multiple data
    plot(t_test, dYp) % Plot predicted signal
    legend(["y_n", "y_{predicted}"]) % Set legend

    subplot(3,1,3) % Create subplot 3
    hold off % Hold off to clear previous plots
    plot(t_test, y0_test) % Plot clean test signal
    title("Clean test signal vs Prediction") % Set title
    hold on % Hold on to plot multiple data
    plot(t_test, dYp) % Plot predicted signal
    legend(["y_0", "y_{predicted}"]) % Set legend

    subplot(3,3,1) % Create subplot 1
    plot(epoch, Loss_test_ideal, '.b', 'MarkerSize', 16) % Plot ideal test loss
    title("Loss_{test ideal} and Loss_{test real} trends along epochs") % Set title
    hold on % Hold on to plot multiple data
    plot(epoch, Loss_test_real, '.r', 'MarkerSize', 16) % Plot real test loss
    set(gca,'YScale','log') % Set y-axis scale to log
    legend(["Loss_{test ideal}", "Loss_{test real}"]) % Set legend

    drawnow % Update figure

    %%
    [Loss_best, ind_ep] = SaveBest_gAE(Loss_best, Loss_test_real, Gemini, gAE, ind_ep, parameters, MSE, Loss_test_ideal, Dkl); % Save the best model

    ind_ep = ind_ep + 1; % Increment epoch index

    
end
