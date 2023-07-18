function [trainmeans, valmeans, testmeans, errors, pred, target, net] = sensorTrain(inp, out, fclayers, ratio, figs)
% Train feedforward neural network

    % INPUTS
    % inp: filtered/extracted sensor data
    % out: [x y depth] targets
    % fclayers: hidden layer sizes - all fully connected, with tanh activation
        % [300 50 20] used throughout paper.
    % ratio: [train val test] ratios: must add to 1
    % figs: plot figures/training progress?

    positions = out(:,1:2);
    inp = normalize(inp);

    %% Scale outputs
    out(:,1) = out(:,1)*10;
    out(:,2) = out(:,2);
    out(:,3) = out(:,3)*10;
    
    %% Extract training, validation, and test data
    %assert(sum(ratio)==1);

    % Training data
    P=randperm(size(inp,1)); % randomise order first to avoid temporal effects
    XTrain=inp(P(1:round(ratio(1)*size(inp,1))),:);
    YTrain=out(P(1:round(ratio(1)*size(inp,1))),:);
    TrainPositions = positions(P(1:round(ratio(1)*size(inp,1))),:);
    
    % Validation data
    XVal=inp(P(round(ratio(1)*size(inp,1))+1:round(sum(ratio(1:2))*size(inp,1))),:);
    YVal=out(P(round(ratio(1)*size(inp,1))+1:round(sum(ratio(1:2))*size(inp,1))),:);
    ValPositions = positions(P(round(ratio(1)*size(inp,1))+1:round(sum(ratio(1:2))*size(inp,1))),:);
    
    % Test data
    XTest=inp(P(round(sum(ratio(1:2))*size(inp,1)+1):end),:);
    YTest=out(P(round(sum(ratio(1:2))*size(inp,1)+1):end),:);
    TestPositions = positions(P(round(sum(ratio(1:2))*size(inp,1)+1):end),:);


    %% Build network architecture and training options
    layers = [featureInputLayer(size(XTrain,2),"Name","featureinput")];
    for i = 1:length(fclayers)
        fcname = "fc"+string(i);
        tanhname = "tanh"+string(i);
        layers = [layers fullyConnectedLayer(fclayers(i),...
            "Name", fcname) tanhLayer("Name",tanhname)];
    end
    layers = [layers fullyConnectedLayer(size(out, 2),"Name","fc_out")...
        regressionLayer("Name","regressionoutput")];
    
    if figs
        plotstring = 'training-progress';
    else
        plotstring = 'none';
    end

    opts = trainingOptions('sgdm', ...
        'MaxEpochs',5000, ...
        'MiniBatchSize', 500,...
         'ValidationData',{XVal,YVal}, ...
        'ValidationFrequency',30, ...
        'GradientThreshold',1000, ...
        'ValidationPatience',1000,...
        'InitialLearnRate',0.05*1, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',500, ...
        'LearnRateDropFactor', 0.99,... %0.05, ...
        'Verbose',0, ...
        'Plots',plotstring, 'ExecutionEnvironment','auto');
    
    %% Train network
    [net, ~] = trainNetwork(XTrain,YTrain,layers, opts);

    %% Calculate and return mean errors for training, validation, and test sets

    [errors, ~, ~] = calculateErrors(XTrain, YTrain, TrainPositions, net, figs);
    if figs
        sgtitle('Train');
    end
    trainmeans = mean(abs(errors));

    [errors, ~, ~] = calculateErrors(XVal, YVal, ValPositions, net, figs);
    if figs
        sgtitle('Validation');
    end
    valmeans = mean(abs(errors));

    [errors, pred, target] = calculateErrors(XTest, YTest, TestPositions, net, figs);
    if figs
        sgtitle('Test');
    end
    testmeans = mean(abs(errors));

end