
%% // --------------------------------------------------------------
%    ***FARYADELL SIMULATION FRAMEWORK***
%    Creator:   Abolfazl Delavar
%    Web:       https://github.com/abolfazldelavar
%% // --------------------------------------------------------------

%% -----------------------------------------------------------------------------
% - RECURSIVE LEAST SQUARE FUNCTION ------------------------------------------
% --------------------------------------------------------------------------
% ---- INSTRUCTION -------------------------------------------------------
% Discrete-time setup: -------------------------------------------------
% 1) Copy the below codes into 'valuation.m':
%        %% Recursive Least Square Parameters
%        params.esType             = 'd';                     %1% 'c' ~> continuous-time, 'd' ~> discrete time
%        params.smootherGain       = 1e+3;                    %2% Smoother gain used to provide theta using covariance
%        params.forgettingfactor   = 1;                       %3% Forgetting factor
%        params.aTheta0            = [-0.5, -0.5];            %4% Initial value of denominator theta
%        params.bTheta0            = -0.5;                    %5% Initial value of nominator theta
%        initialCovarianceValue    = 1e+3;                    %6% Initial parameter covariance
%        params.numaTheta          = numel(params.aTheta0);
%        params.numbTheta          = numel(params.bTheta0);
%        params.P0                 = eye(params.numaTheta + params.numbTheta)*initialCovarianceValue;
% 
% 2) Copy the below codes into 'initialization.m':
%        %% Recursive least square signals
%        numTheta                  = params.numaTheta + params.numbTheta;
%        models.theta              = [[params.aTheta0(:); params.bTheta0(:)], zeros(numTheta, params.n)];
%        models.smootherTheta      = models.theta;
%        models.P                  = zeros(numTheta, numTheta, params.n + 1);
%        models.P(:,:,1)           = params.P0;
% 
% 3) Copy the below codes into the main loop in 'simulation.m', where you want to estimate parameters:
%        % Estimation plant parameters
%        models                    = caller.recursiveLeastSquare(params, models, func, t);
% 
% 4) Set the 'signalInput', 'signalOutput', and 'delaySystem' values at this file.
% 
%
% Continuous-time setup: --------------------------------------------------
% 1) Copy the below codes into 'valuation.m':
%        %% Recursive Least Square Parameters
%        params.esType             = 'c';                     %1% 'c' ~> continuous-time, 'd' ~> discrete time
%        params.smootherGain       = 1e+3;                    %2% Smoother gain used to provide theta using covariance
%        params.forgettingfactor   = 1;                       %3% Forgetting factor
%        params.aTheta0            = [-0.5, -0.5];            %4% Initial value of denominator theta
%        params.bTheta0            = -0.5;                    %5% Initial value of nominator theta
%        initialCovarianceValue    = 1e+3;                    %6% Initial parameter covariance
%        % The order of below filter must be higher than max(Order a, Order b)
%        params.esFilter           = [1, 2, 1];               %7% prefiltering input and output signals
%        params.numaTheta          = numel(params.aTheta0);
%        params.numbTheta          = numel(params.bTheta0);
%        params.P0                 = eye(params.numaTheta + params.numbTheta)*initialCovarianceValue;
% 
% 2) Copy the below codes into 'initialization.m':
%        %% Recursive least square signals
%        numTheta                  = params.numaTheta + params.numbTheta;
%        theta0Biased              = [params.aTheta0(:); params.bTheta0(:)];
%        models.theta              = [theta0Biased, zeros(numTheta, params.n)];
%        models.smootherTheta      = models.theta;
%        models.P                  = zeros(numTheta, numTheta, params.n + 1);
%        models.P(:,:,1)           = params.P0;
%        for i = 1:params.numaTheta
%            esFilter              = tf([1 zeros(1, i-1)], params.esFilter);
%            models.esaFilter{i}   = linearSystem(esFilter, params.step, models.Tline);
%        end
%        for i = 1:params.numbTheta
%            esFilter              = tf([1 zeros(1, i-1)], params.esFilter);
%            models.esbFilter{i}   = linearSystem(esFilter, params.step, models.Tline);
%        end
% 
% 3) Copy the below codes into the main loop in 'simulation.m', where you want to estimate parameters:
%        % Estimation plant parameters
%        models                    = caller.recursiveLeastSquare(params, models, func, t);
% 
% 4) Set the 'signalInput', 'signalOutput', and 'delaySystem' values to this file.
% 
%% ------------------------------------------------------------------------------
function models = recursiveLeastSquare(params, models, func, t)
                        
    signalInput     = models.yourInputSignal;            % Input signal (full time)
    signalOutput    = models.yourOutputSignal;           % Output signal (full time)
    delaySystem     = models.yourDelayTime;              % delay of the system (plant)
    
    switch params.esType
        case 'd'    % Discrete-time recursive least square
            [models.theta(:,t+1),                           ...
             models.P(:,:,t+1)] = dtrls(params,             ...        % Import parameters
                                        func,               ...        % Import useful functions
                                        signalInput,        ...        % Input signal
                                        signalOutput,       ...        % Output signal
                                        delaySystem,        ...        % Plant delay value
                                        models.theta(:,t),  ...        % Last parameters
                                        models.P(:,:,t),    ...        % Last parameters covariance
                                        t);                            % current sample time
        case 'c'    % Continuous-time recursive least square
            models = ctrls(params,                                      ...     % Import parameters
                            models,                                     ...     % Import models
                            func.delayed(signalInput, t, delaySystem),  ...     % Last delayed input
                            signalOutput(:,t),                          ...     % Last output
                            t);                                                 % current sample time
    end
    
    % Smoother part: select the best choice theta between theta0 and estimated value
    % due to that information matrix (inverse of covariance called information).
    informationMatrix           = (models.P(:,:,t+1))^-1;
    numTheta                    = params.numaTheta + params.numbTheta;
    models.smootherTheta(:,t+1) = ((informationMatrix + params.smootherGain*eye(numTheta))^-1)*    ...
                                   (informationMatrix*models.theta(:,t+1) +                        ...
                                   params.smootherGain*eye(numTheta)*models.smootherTheta(:,1));
end

%% Continuous-Time Recursive Least Square
function models = ctrls(params, models, u, y, t)
    
    % Making the regressor by inserting filtered data
    Phi = zeros(params.numaTheta + params.numbTheta, 1);
    for i = 1:params.numaTheta
        models.esaFilter{i} = models.esaFilter{i}.nextstep(y);
        Phi(params.numaTheta - i + 1) = models.esaFilter{i}.outputs(t);
    end
    for i = 1:params.numbTheta
        models.esbFilter{i} = models.esbFilter{i}.nextstep(u);
        Phi(params.numaTheta + params.numbTheta - i + 1) = models.esbFilter{i}.outputs(t);
    end
    
    % Pre-shift Theta to prepare for Identification part
    preFilterBiase      = params.esFilter(1 + (1:params.numaTheta));
    lastTheta           = [preFilterBiase(:) - models.theta(1:params.numaTheta, t) ;
                           models.theta((1:params.numbTheta) + params.numaTheta,t)];
    
    % The main code area that estimates using continuous RLS
    lastP               = models.P(:,:,t);
    errsig              = y - Phi'*lastTheta;                   % error signal between estimator and plant
    K                   = lastP*Phi;                            % Estimator Gain
    PhiPhi              = Phi*Phi';
    newTheta            = lastTheta + params.step*(K*errsig);   % Update new theta
    models.P(:,:,t+1)   = lastP     + params.step*(params.forgettingfactor*lastP -(lastP*PhiPhi*lastP));
                                                                % Update covariance matrix
    
    % Calculating the real parameters and saving them
    models.theta(:,t+1) = [preFilterBiase(:) - newTheta(1:params.numaTheta)  ;
                           newTheta((1:params.numbTheta) + params.numaTheta)];
end

%% Descrete-Time Recursive Least Square
function [Theta, newP] = dtrls(params, func, signalInput, signalOutput, delaySystem, lastTheta, lastP, t)
    
    % This part of code are seeking to provide 'y' to use in line 155
    y = zeros(1, params.numaTheta);
    for i = 1:params.numaTheta
        y(i) = func.delayed(signalOutput, t, i);
    end
    
    % This part of code are seeking to provide 'u' to use in line 155
    u = zeros(1, params.numbTheta);
    for i = 1:params.numbTheta
        u(i) = func.delayed(signalInput, t, delaySystem + i);
    end
    
    lastOutput = signalOutput(t);
    Phi        = [-y, u]';                                                 % Making regressor
    eps        = lastOutput - Phi'*lastTheta;                              % Error signal between estimator and plant
    K          = (lastP*Phi)/(params.forgettingfactor + Phi'*lastP*Phi);   % Estimator Gain
    PhiPhi     = Phi*Phi';
    Theta      = lastTheta + K*eps;                                        % Updating new theta
    newP       = (lastP - (lastP*PhiPhi*lastP)/(params.forgettingfactor + Phi'*lastP*Phi))/params.forgettingfactor;
                                                                           % Updating covariance matrix
end
