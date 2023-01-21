function [Yn On Yt Ot wt] = ANN1912705(lr, N_ep, tsf, af, pindx)
% lr is the learning rate, N_ep the number of epochs, tsf training size fraction
% af = 0 for sigmoid code, af not zero for ReLU code
% Y/O are the exact/predicted labels/targets (n=train, t=test); wt is test success
% MNIST CSV files not to be altered, and in the same folder as matlab code.
... YOUR CODE GOES HERE ...
	%%% Information %%%
	% Your Values Are:
	% u = 26, v = 33, w = 33
	% D = 2
	% Learning Rate, alpha = lr = 0.0126
 	% Number of Epochs, N_ep = 333

	% Initialise u v and w
	u = 26; v = 33; w = 33;

	%%% Check that Inputs are valid
	if lr <= 0
		error("Learning rate cannot be less than or equal to 0");
	end
	if N_ep <= 0
		error("Number of Epochs has to be greater than 0");
	end
	if tsf <=0 || tsf > 1
		error("Training size fraction has to be greater than 0, and less than or equal to 1");
	end
	
	% Determine activation function
	if af == 0 % Sigmoid
		f = @(x) 1./(1+exp(-x));
		df = @(x) exp(-x)./((1+exp(-x)).^2);
	else % ReLU
		f = @(x) max(0,x);
		df = @(x) 1*(x>=0);
	end	

	% Declare softmax function
	softmax = @(x)(1./sum(exp(x))).*exp(x);
	
	% For the cost function: TSE = 0, XE = 1, 
	cf = pindx;

	% Load in the CSV data
	trainSet = readmatrix("MNIST_train_1000.csv");
	testSet = readmatrix("MNIST_test_100.csv");

	% Get size of Training and Test sets
	nTrain = length(trainSet(:, 1));
	nTest = length(testSet(:, 1));

	% Set the Neural Network layer sizes
	layernums = [784 u v w 10];
	numLayers = length(layernums);
	
	oldNTrain = nTrain; % to keep for mixups
	nTrain = oldNTrain*tsf; 
	trainSet(:, 2:end) = trainSet(:,2:end)/255; % pixel value should be 0-1, so sigmoid doesn't cut it off
	testSet(:,2:end) = testSet(:,2:end)/255;

	for i = 2:numLayers
		% Choose weights depending on Activation function
		if af == 0 % Sigmoid can have negative weights
			a = 2;
			layer(i).W = a*rand(layernums(i-1), layernums(i)) - a/2; % Weights between layer i-1 -> i 	
		else % With 5 layers, I wonder if the ouptut by the 4th layer is mostly 0s, that would be unhelful, so let's make all weights positive.
			% using "He initialisation"
			% https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
			a = (sqrt(2/layernums(i-1)));
			layer(i).W = 2*a*rand(layernums(i-1), layernums(i)) - a;
		end

		layer(i).B = zeros(layernums(i), 1);%a*rand(layernums(i), 1) - a/2; % Biases on layer i	
		layer(i).n = zeros(layernums(i), 1); % unactivated outputs
		layer(i).a = zeros(layernums(i), 1); % activated outputs
		layer(i).S = zeros(layernums(i), 1); % S value
	end
	

	% Train the Neural Network
	mixup = randperm(oldNTrain); % Mixup all the values in the complete dataset
	values = mixup(1:nTrain); % Values I'll be working with

	for epoch = 1:N_ep % for N_ep epochs
		values = values(randperm(nTrain));
		for j = 1:nTrain % for each Training point
			
			% Get a random index
			i = values(j);

			% Get the input to the Neural Network
			x = trainSet(i, 2:785)';	% a bit of hard code		

			% Get the true ouput from now
			t = zeros(10,1); t(trainSet(i,1) + 1) = 1;

			%% Forward Propagate
			% first for the first input since it's easier outside the for loop
			layer(2).n = layer(2).W'*x + layer(2).B; layer(2).a = f(layer(2).n);

			% now for the rest of the layers
			for k = 3:numLayers
				layer(k).n = layer(k).W'*layer(k-1).a + layer(k).B; % Get Unactivated output
				layer(k).a = f(layer(k).n); % Get Activated output
			end

			% Change the final activation function to softmax if Cross Entropy (XE) 
			if cf == 1
				layer(numLayers).a = softmax(layer(numLayers).n);		
			end
			
			% Calculate Error
			e = t - layer(numLayers).a;

			% Get the first Value of S,
			if cf == 0 % TSE  
				A = diag(df(layer(numLayers).n)); % Get the A matrix (diagonal derivatives)
				layer(numLayers).S = -2*A*e; % Get the S value at the last layer
			else % Cross Entropy (XE)
				layer(numLayers).S = -e;	
			end

			% Calculate S values with Calculus Bases Backprop
			for k = numLayers-1:-1:2
				layer(k).S = diag(df(layer(k).n))*layer(k+1).W*layer(k+1).S; % SAWS 
			end	

			%% Update the values of Weights and Biases
			% first for the first set of weights (layer 2)
			layer(2).W = layer(2).W - lr*x*layer(2).S'; % Weights Update
			layer(2).B = layer(2).B - lr*layer(2).S; % Biases Update

			% Then for the rest of the layers (3,4,5)
			for k = 3:numLayers
				layer(k).W = layer(k).W - lr*layer(k-1).a*layer(k).S'; % Weights Update
				layer(k).B = layer(k).B - lr*layer(k).S; % Biases Update
			end	
		end % of train set
	end % of training epoch
	
	
	%%% Test the Neural Network
	
	% Make a confusion matrix to store values
	correct = 0;
	
	% Initalise function outputs Yn and On
	Yn = zeros(10,nTrain);	
	On = zeros(10,nTrain);


	% Test the Neural Network on the training data
	for j = 1:nTrain
		 
		% Get a random index
		i = values(j);

		% Get the input to the Neural Network
		x = trainSet(i, 2:785)';	% a bit of hard code		

		% Get the true ouput from now
		t = zeros(10,1); t(trainSet(i,1) + 1) = 1;
		Yn(:,j) = t;

		%% Forward Propagate
		% first for the first input since it's easier outside the for loop
		layer(2).n = layer(2).W'*x + layer(2).B; layer(2).a = f(layer(2).n);

		% now for the rest of the layers
		for k = 3:numLayers
			layer(k).n = layer(k).W'*layer(k-1).a + layer(k).B; % Get Unactivated output
			layer(k).a = f(layer(k).n); % Get Activated output
		end

		% Change the final activation function to softmax if Cross Entropy (XE) 
		if cf == 1
			layer(numLayers).a = softmax(layer(numLayers).n);		
		end

		% make the output one hot
		out = zeros(10,1);

		% get the hottest index
		[~ , ind] =  max(layer(numLayers).a);
		
		% set out variable
		out(ind) = 1;
		On(:,j) = out;
		
	end

	
	%% Test the Neural Network on test data
	% initialise Yt and Ot for the function outputs
	Yt = zeros(10,nTest);
	Ot = zeros(10,nTest);
	 
	for i = 1:nTest
		%% Copy of Forward propagation from earlier, altered for the testing data

		% Get the input to the Neural Network
		x = testSet(i, 2:785)';	% a bit of hard code		
		
		% Get the true ouput from now
		t = zeros(10,1); t(testSet(i,1) + 1) = 1;

		% Populate Yt
		Yt(:,i) = t;
		
		% first for the first input since it's easier outside the for loop
		layer(2).n = layer(2).W'*x + layer(2).B; layer(2).a = f(layer(2).n);

		% now for the rest of the layers
		for k = 3:numLayers
			layer(k).n = layer(k).W'*layer(k-1).a + layer(k).B; % Get Unactivated output
			layer(k).a = f(layer(k).n); % Get Activated output
		end

		% Change the final activation function to softmax if Cross Entropy (XE)
		if cf == 1
			layer(numLayers).a = softmax(layer(numLayers).n);		
		end

		% make the output one-hot/get the hottest index from the NN
		[~ , ind] =  max(layer(numLayers).a);
		
		%Populate Ot
		Ot(ind, i) = 1;

		% Check if correct prediction
		actual = testSet(i,1);
		found = ind-1;

		% Get number of successful classifications
		if actual == found
			correct = correct + 1;
		end
	end
	
	% Get number of correct classifications for the specific run and tsf (as a Percentage)
	wt = correct*100/nTest;

end

