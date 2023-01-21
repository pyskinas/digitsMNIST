clc; clear all; close all; format long; 


% set some useful defaults
set(0,'DefaultLineLineWidth', 2);
set(0,'DefaultLineMarkerSize', 10);

% % Choose the activation function
% af = 0;
% 
% % one for TSE, one for XE 
% for cf = 0:1
% 		
% 	% keep track of successes so that you can do statistical analysis on it later
% 	% rows =  tsf value
%  	% columns = run
% 	success = zeros(4);
% 
% 	% Get a random Training size Fraction (tsf) from the training set
% 	% Need to be able to get the index of the tsf
% 	tsfs = [0.01 0.1 0.5 1];
% 
% 	for tsfInd = 1:4 % for different sizes of training sets	
% 		% set tsf
% 		tsf = tsfs(tsfInd);
% 
% 		for run = 1:4 % told to do 4 runs to get averages
% 			fprintf("Cost function %d, TSF of %.0f, run %d\n", cf, tsf, run);
% 			[~,~,~,~, success(tsfInd, run)] =  ANN1912705(0.0126, 333, tsf, af, cf);
% 		end % of run
% 	end % of tsfs
% 
% 	% Plot successes against training set size 0 for TSE, 1  for XE
% 	if cf == 0 % TSE in RED
% 
% 		% success average
% 		sav0 = mean(success');
% 
% 		% Success lower bound (minus 1 standard deviation)
% 		slb0 = sav0 - std(success');
% 
% 		% Success upper bound (plus 1 standard deviation)
% 		sub0 = sav0 + std(success');
% 
% 	else % XE in BLUE
% 
% 		% success average
% 		sav1 = mean(success');
% 
% 		% Success lower bound (minus 1 standard deviation)
% 		slb1 = sav1 - std(success');
% 
% 		% Success upper bound (plus 1 standard deviation)
% 		sub1 = sav1 + std(success');
% 
% 	end
% 
% 	toPlot = 1;
% 	if toPlot % to plot the percentage successes
% 		hold on
% 		%% Plot the percentage successes
% 		% first plot averages to get a nice legend
% 		semilogx(tsfs*1000, sav0, "rd-", "LineWidth", 2); % TSE
% 		semilogx(tsfs*1000, sav1,  "b+-", "LineWidth", 2); % XE
% 
% 		% TSE
% 		semilogx(tsfs*1000, slb0, "rd:", "LineWidth", 1);
% 		semilogx(tsfs*1000, sub0,"rd:", "LineWidth", 1);
% 
% 		% XE
% 		semilogx(tsfs*1000, slb1,  "b+:", "LineWidth", 1);
% 		semilogx(tsfs*1000, sub1, "b+:", "LineWidth", 1);
% 
% 		xlabel("Training Set Size");
% 		ylabel("Accuracy (%)");
% 		set(gca, 'XScale', 'log'); % Make it logarithmic
% 		axis([10 1000 0 100]);
% 
% 		if af == 0
% 		% Add Labels to graph
% 			title("Sigmoid Success Percentages for 1912705");
% 			legend("TSE with Sigmoid", "XE with Sigmoid", "Location", "NW");
% 		else
% 			title("ReLU Success Percentages for 1912705");
% 			legend("TSE with ReLU", "XE with ReLU", "Location", "NW");
% 		end
% 		hold off
%     end
% end % of cost functions

% Start of Confusion Matrices, questions 5 and 6
alpha = 0.0126;
N_ep = 333;
tsf = 0.2;
af = 0;
cf = 0;
[Yn On Yt Ot wt] = ANN1912705(alpha, N_ep, tsf, af, cf); 

% set up some variables 
nTrain = length(Yn); % number of Training data points 
nTest = length(Yt); % number of Test data points
Yd = zeros(2, nTest); % how many data points are D = 2; (1 0) for D, (0 1) for not D
Od = zeros(2, nTest); % outpust for whether D or not;   

for i = 1:nTest  
    if Yt(3,i) == 1
        Yd(1,i) = 1;
    else
        Yd(2,i) = 1; 
    end

    if Ot(3,i) == 1
        Od(1,i) = 1;
    else
        Od(2,i) = 1;
    end
end    

% Plot the confusion matrices side by side 
hold on

% plot for TSE
subplot(1,2,1);
plotconfusion(Yd, Od, "ReLU detecting '2' with TSE"); 
set(findobj(gca,'type','text'),'fontsize',20)

% Refind values for XE
[Yn On Yt Ot wt] = ANN1912705(alpha, N_ep, tsf, af, 2); 
Yd = zeros(2, nTest); % how many data points are D = 2; (1 0) for D, (0 1) for not D
Od = zeros(2, nTest); % outpust for whether D or not;   

for i = 1:nTest  
    if Yt(3,i) == 1
        Yd(1,i) = 1;
    else
        Yd(2,i) = 1; 
    end

    if Ot(3,i) == 1
        Od(1,i) = 1;
    else
        Od(2,i) = 1;
    end
end 

% plot with XE
figure 
subplot(1,2,2);
plotconfusion(Yd, Od, "ReLU detecting '2' with XE"); 
set(findobj(gca,'type','text'),'fontsize',20)
hold off 
