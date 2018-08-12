clear; clc;
start=0.5; step=0.1; tot_steps=2;
allres=zeros(tot_steps,1);
for i=1:tot_steps
    templam=start+step*(i-1);
    temp=KL_2(templam);
    allres(i)=max(temp);
end