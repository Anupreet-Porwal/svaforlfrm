function [z]=opti_z_old(z,w,train,lambda2)
    n=size(z,1);
    curr=ziloptimiser_opti(z,z(1,1),w,train,1,1,lambda2);
    for i=1:n  
        K=size(z,2);
        k=1;
        while(k<=K)
            zprop=1-z(i,k);            
            prop=ziloptimiser_opti(z,zprop,w,train,i,k,lambda2);
            if (prop<curr)
                z(i,k)=zprop;
                curr=prop;
                did_we_lose=(sum(z(:,k)))==0;
                if(did_we_lose)
                    z(:,k)=[];
                    w(k,:)=[]; w(:,k)=[];
                    K=K-1;
                    k=k-1;
                end
            end
            k=k+1;
        end
    end
end