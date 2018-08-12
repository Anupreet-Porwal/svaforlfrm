function [z,w]=opti_z_new(z,w,train,lambda2)
    n=size(z,1);C=size(z,2);
    Pleft=z*w; Pright=w*z'; Pval=Pleft*z';
    premask = ~isnan(train);
    train(isnan(train))=0;    
    zobj_old = (-1*sum(sum(train.*Pval))) + sum(sum(premask.*(log(1+exp(Pval))))) + (C)*lambda2;
    for i=1:n
        K=size(z,2);
        k=1;
        while(k<=K)
            T=zeros(n,n);
            T(i,:)=T(i,:)+Pright(k,:);
            T(:,i)=T(:,i)+Pleft(:,k);
            zdiff=(1-z(i,k))-z(i,k); %znew-zold;
            T=zdiff*T;
            T(i,i)=T(i,i)+w(k,k);  
            did_we_lose=(sum(z(:,k))-z(i,k))==0;
            zobj_new = (-1*sum(sum(train.*(Pval+T)))) + sum(sum(premask.*(log(1+exp(Pval+T))))) + (K-did_we_lose)*lambda2;
            if(zobj_new<zobj_old) % the case where we change the value and hence update our carriers
                z(i,k)=1-z(i,k);
                zobj_old=zobj_new;
                Pright(:,i) = Pright(:,i) + zdiff*w(:,k);
                Pleft(i,:) = Pleft(i,:) + zdiff*w(k,:);
                Pval=Pval + T;
                if(did_we_lose)
                    z(:,k)=[];
                    w(k,:)=[]; w(:,k)=[];
                    Pleft=z*w; Pright=w*z'; Pval=Pleft*z';
                    premask = ~isnan(train);
                    train(isnan(train))=0;    
                    zobj_old = (-1*sum(sum(train.*Pval))) + sum(sum(premask.*(log(1+exp(Pval))))) + (C)*lambda2;
                    K=K-1;
                    k=k-1;
                end
            end
            k=k+1;            
        end
    end
end