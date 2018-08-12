% MAD LFRM 
% April 23, 2017 
clear;
clc;
%% dataset jazz
d=load('lazega-lawyers.mat'); 
data=d.A_adv;   %loading the dataset
N=length(data);
tr_ratio=0.8;
lambda2=0.5;
train=NaN(N);
test=NaN(N);
score=NaN(N);
C=1;
%% Initializations
z = randi([0,1],N,C); %Z = [N,C] random binary matrix
%Initiazlise w as cxc matrix
w=normrnd(0,1,C,C);
nsim=30;
% Form training and test matrix and compute AUC score using random initialisation of Z and w to compare
Pval=sigmf((z*w)*(z'));
for i=1:N
    p=randperm(N,round(tr_ratio*N));
    tf=false(N,1);
    tf(p) = true;
    train(i,tf)=data(i,tf);
    test(i,~tf)=data(i,~tf);
    score(i,:)=(~tf)'.*Pval(i,:);
    score(i,tf)=NaN;
    train(i,i)=NaN;
    test(i,i)=NaN;
    score(i,i)=NaN;
end
[X,Y,T,AUC_old]=perfcurve(test(~isnan(test)), score(~isnan(score)), 1);
%%
eps=10^(-3);
%Qold=Qobj(w,train,z,lambda2)
Qold = Qobj_opti(w,train,z,lambda2);
for sim=1:nsim     % The opti loop
%optimise wrt Z like in FL means algo by choosing an optimal choice of a particular row of Z from all possible choices and keeping other row fixed
    zposs=de2bi(0:(2^C-1));
    for i=1:N
        sum=zeros(length(zposs),1);
        for l=1:length(zposs)
            sum(l)=zoptimiser_opti(z,zposs(l,:),w,train,i);
        end
        [M,id]=min(sum);
        z(i,:)=zposs(id,:);
    end
%     Qnew=Qobj_opti(w,train,z,lambda2);
% %    Suppose after one iteration over all the points I get updated z 
%    % [test_r,test_c]=find(~isnan(test));
%    Pvalnew=sigmf((z*w)*z');
%    testnanmask=test;
%    testnanmask(~isnan(testnanmask))=1;
%    scorenew=testnanmask.*Pvalnew;
%    [X,Y,T,AUC_new]=perfcurve(test(~isnan(test)), score(~isnan(score)), 1);
%    cond=(Qold-Qnew)>eps; %Condition based on objective function evaluation on training dataset to understand whether the optimisation over Z has converged 
%     if(cond==1)
%         Qold=Qnew;
%     end
 
  % optimise wrt W using fminsearch or something of that sort
  wmin=fminsearch('Qobj_opti', w, [],train,z,lambda2);
  w=wmin;
  s=randperm(N,1);
  znew=[z,zeros(N,1)]; % Create Z with an additional feature 
  znew(s,C+1)=1;
  w_c=normrnd(0,1,C,1);
  w_c_t=normrnd(0,1,1,C);
  w_s=normrnd(0,1,1,1);
  wnew=[wmin,w_c;w_c_t,w_s];
  wminnew=fminsearch('Qobj_opti', wnew, [],train,znew,lambda2); % optimise over  Wnew
cond2=1;
count2=1;
%Qold2=Qobj(wminnew,train,znew,lambda2);
 % while cond2==1 % optimise over Znew given wminnew
        %count2=count2+1
        zposs2=de2bi(0:(2^(C+1)-1));
        for i=1:N
            sum=zeros(length(zposs2),1);
            for l=1:length(zposs2)
                sum(l)=zoptimiser_opti(znew,zposs2(l,:),wminnew,train,i);
            end
            [M2,id2]=min(sum);
            znew(i,:)=zposs2(id2,:); 
        end
        Qnew2=Qobj_opti(wminnew,train,znew,lambda2);
    %Suppose after one iteration over all the points I get updated z 
%     scorenew=NaN(n);
%    % [test_r,test_c]=find(~isnan(test));
%     for m=1:n
%         for k=1:n
%            if(~isnan(test(m,k)))
%                scorenew(m,k)=sigmf(z(m,:)*w*z(k,:)',[1 0]);
%            end
%         end
%     end
%     
%     [X,Y,T,AUC_new]=perfcurve(test(~isnan(test)), score(~isnan(score)), 1);
%     AUC_new
%    cond2=(Qold2-Qnew2)>eps;
%     if(cond2==1)
%         Qold2=Qnew2;
%     end
  %end
  % find W optimising the same
  Lold=Qobj_opti(wmin,train,z,lambda2);
  Lnew=Qobj_opti(wminnew,train,znew,lambda2);
  if (Lnew<Lold)  % Calc Q for two sets of values and choose the one that minimises the
      C=C+1      % objective function
      z=znew;
      w=wminnew;
  end
  %Just to keep check of how the algorithm is performing
  Pvalnew=sigmf((z*w)*z');
   testnanmask=test;
   testnanmask(~isnan(testnanmask))=1;
   scorenew=testnanmask.*Pvalnew;
    
    [X,Y,T,AUC_new]=perfcurve(test(~isnan(test)), scorenew(~isnan(scorenew)), 1);
    AUC_new_vec(sim)=AUC_new
end

    