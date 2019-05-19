tic
filename='C:\Users\avina\OneDrive\Documents\final project\pmtk3-master\pmtk3-master\fisherIrisVersicolorVirginicaData.txt';
data=load(filename); %loading data
%data = data(randperm(size(data,1)),:);
x=data(1:end,1:4); %making first two features into imput variable x
y=data(1:end,5); % making last feature into output variable y
xtest = data(81:end,1:4);
ytest = data(81:end,5);
Xtest = [ones(length(xtest), 1) xtest];
X = [ones(length(x), 1) x]; % adding 1's inorder to match with the linear regression parameters
coefficients = zeros(5, 1); %initializing coefficients to 0's



learningRate =0.1; %initializing learning rate
fprintf('learning rate: ');
disp(learningRate);

function [ value ] =sigmoid(m,coefficients)
  value= 1./(1+exp(-(m*coefficients)));
end



function [ cost_cal ] = costfunction( cal_X, y, coefficients )
    %   Calculates the cost function
    hat=sigmoid(cal_X , coefficients);
    cost_cal = (-1)*(y'*log(hat)+(1-y)'*log(1-hat)); % Calculating Cost Function 
end

function [coefficients ] = descent(each_X,each_y,coefficients,learningRate)
  change=(each_X'*(sigmoid(each_X,coefficients)-each_y));
  coefficients=coefficients -learningRate*change;
end

function [ coefficients, costHistory, iterations ] = sgd( X, y,learningRate)
    coefficients=zeros(5,1); 
    len = length(y);
    noc = length(coefficients); % number of coefficients
    % Creating a vector of zeros for storing our cost function history
    costHistory(1) = 0;
    cost=0;
    
    flag = 1;
    i=2; 
    
    while flag   
      for k=1:len
        coefficients= descent(X(k,:),y(k),coefficients,learningRate);
        cost_new = costfunction(X(k,:),y(k),coefficients);
        cost=cost+cost_new;
      end
      cost=cost/length(y);
      costHistory(i)=cost;
      cost=0;
        
      if (abs(costHistory(i)-costHistory(i-1))<=0.0000001) 
        flag=0;
        iterations=i;
        break
      end
        i++;
     end
end 
function [ result ] = logclassify(Xpredict,coefficients)
  result=[];
  hatnew=sigmoid(Xpredict,coefficients);
  n=length(Xpredict);
  for i=1:n
    if hatnew(i)>=0.5   %decision boundary
      hatnew(i)=1;
    else
      hatnew(i)=0;
    end
  end
  result=hatnew;
end

%calling gradient function   
[coefficients, costHistory, iterations] = sgd(X, y, learningRate); 
result=logclassify(Xtest,coefficients);
trainpredict=logclassify(X,coefficients);
trainerror=mean(trainpredict ~= y);
testerror = mean(result ~= ytest);
testaccuracy = (1-testerror)*100;
trainaccuracy = (1-trainerror)*100;

fprintf('\nthe coefficients are:\n');
disp(coefficients);  %displaying coefficients
fprintf('the number of iterations to converge: ');
disp(iterations); % displaying number of iterations

fprintf('Accuracy on test data: %.2f%%\n',testaccuracy);
fprintf('Accuracy on train data: %.2f%%\n',trainaccuracy);

   
toc