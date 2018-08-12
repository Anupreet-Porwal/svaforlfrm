function [AUC_new_vec]=KL_2(lambda2)
%% Dataset Jazz
d=load('nips234.mat'); 
data=d.B;   %loading the dataset
n=length(data);
tr_ratio=0.8;
%lambda2=0.5;
train=NaN(n);
test=NaN(n);
score=NaN(n);
C=1;
%% Initializations
z = randi([0,1],n,C); %Z = [N,C] random binary matrix
w=normrnd(0,1,C,C); %Initiazlise w as cxc matrix
nsim=10;
% Form training and test matrix and compute AUC score using random initialisation of Z and w to compare
Pval=sigmf((z*w)*(z'));
for i=1:n
    p=randperm(n,round(tr_ratio*n));
    tf=false(n,1);
    tf(p) = true;
    train(i,tf)=data(i,tf);
    test(i,~tf)=data(i,~tf);
    score(i,:)=(~tf)'.*Pval(i,:);
    score(i,tf)=NaN;
    train(i,i)=NaN;
    test(i,i)=NaN;
    score(i,i)=NaN;
end
premask = ~isnan(train);
train_san=train;
train_san(isnan(train_san))=0;
[~,~,~,AUC_old]=perfcurve(test(~isnan(test)), score(~isnan(score)), 1);
testnanmask=test;
testnanmask(~isnan(testnanmask))=1;
%%
for sim=1:nsim     
    %optimise wrt Z like in FL means algo by choosing an optimal choice of a particular row of Z from all possible choices and keeping other row fixed
  [z,w]=opti_z_new(z,w,train,lambda2);      
  % optimise wrt W using fminsearch or something of that sort
  wmin=fminsearch('Qobj_extra', w,[],train_san,z,lambda2,premask);
  w=wmin;
  s=randi([1,n]);  
  znew=[z,zeros(n,1)]; % Create Z with an additional feature 
  K=size(z,2);
  znew(s,K+1)=1;
  w_c=normrnd(0,1,K,1);
  w_c_t=normrnd(0,1,1,K);
  w_s=normrnd(0,1,1,1);
  wnew=[wmin,w_c;w_c_t,w_s];
  wminnew=fminsearch('Qobj_extra', wnew, [],train_san,znew,lambda2,premask); % optimise over  Wnew
    % again optimize Z wrt to W
        %[znew]=opti_z_old(znew,wminnew,train,lambda2);
     [znew,wminnew]=opti_z_new(znew,wminnew,train,lambda2);
  % find W optimising the same
  Lold=Qobj_extra(wmin,train_san,z,lambda2,premask);
  Lnew=Qobj_extra(wminnew,train_san,znew,lambda2,premask);
  if (Lnew<Lold)  % Calc Q for two sets of values and choose the one that minimises the
      size(wminnew)
      z=znew;
      w=wminnew;
  end
  %Just to keep check of how the algorithm is performing
  Pvalnew=sigmf((z*w)*z');   
  scorenew=testnanmask.*Pvalnew;
    
    [~,~,~,AUC_new]=perfcurve(test(~isnan(test)), scorenew(~isnan(scorenew)), 1);
    AUC_new_vec(sim)=AUC_new
end
end
% plot(X,Y)
% xlabel('False positive rate')
% ylabel('True positive rate')
% title('ROC for Link Prediction by K-LAFTER-II algorithm for Lazega Lawyers test dataset')
% xlswrite('Feature_matrix_Lazega_K_LAFTER_II.xlsx', z,'Sheet1');
% auc=[X,Y];
%xlswrite('AUC_TPR_FPR_K_LAFTER2_lambda05_lazega.xlsx', auc,'Sheet1');
