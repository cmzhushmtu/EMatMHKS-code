function [u_border,v_border,v0,times_for_train]=trainmethod(cluster_k,cluster_j,c,b,u,v,p,d1,d2,lambda,eta,v0)

times_for_train_max=10000;times_for_train=1;
cluster_all_for_this_classifier=cell(1,3);
cluster_all_for_this_classifier{1,1}='class label£¬1 or -1';
cluster_all_for_this_classifier{1,2}='order of cluster';
cluster_all_for_this_classifier{1,3}='samples';
%get samples of cluster_k
number_of_cluster_k=0;
for number_of_cluster_k_order=1:size(cluster_k,1)
    number_of_cluster_k=number_of_cluster_k+size(cluster_k{number_of_cluster_k_order,1},1);
end
%get samples of cluster_j
number_of_cluster_j=0;
for number_of_cluster_j_order=1:size(cluster_j,1)
    number_of_cluster_j=number_of_cluster_j+size(cluster_j{number_of_cluster_j_order,1},1);
end

row=1;
for size_k_cluster=1:size(cluster_k,1)
    cluster_all_for_this_classifier_for_it=cluster_k{size_k_cluster,1};
    for cluster_all_for_this_classifier_for_it_order=1:size(cluster_all_for_this_classifier_for_it,1)
        cluster_all_for_this_classifier{row,1}=cluster_all_for_this_classifier_for_it(cluster_all_for_this_classifier_for_it_order,1);
        cluster_all_for_this_classifier{row,2}=1;%label of positive samples
        cluster_all_for_this_classifier{row,3}=cluster_k{size_k_cluster,2};
        row=row+1;
    end
end
for size_j_cluster=1:size(cluster_j,1)
    cluster_all_for_this_classifier_for_it=cluster_j{size_j_cluster,1};
    for cluster_all_for_this_classifier_for_it_order=1:size(cluster_all_for_this_classifier_for_it,1)
        cluster_all_for_this_classifier{row,1}=cluster_all_for_this_classifier_for_it(cluster_all_for_this_classifier_for_it_order,1);
        cluster_all_for_this_classifier{row,2}=-1;%label of negative samples
        cluster_all_for_this_classifier{row,3}=cluster_j{size_j_cluster,2};
        row=row+1;
    end
end
u_to_use=u*ones(d1,1);%vectorize of u
v_to_use=v*ones(d2,1);
b_to_use=b*ones(size(cluster_all_for_this_classifier,1),1);
eta_to_use=eta;
v0_to_use=v0;


%get Y
Y=zeros(size(cluster_all_for_this_classifier,1),d2+1);

for row_Y=1:size(cluster_all_for_this_classifier,1)
    yn=cluster_all_for_this_classifier{row_Y,2}*[u_to_use'*cell2mat(cluster_all_for_this_classifier{row_Y,1}),1]';
    Y(row_Y,:)=yn';
end
v_to_use_1=[v_to_use',v0_to_use]';
e_for_compute=(Y*v_to_use_1-ones(size(cluster_all_for_this_classifier,1),1)-b_to_use);
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%update b
b=b_to_use+p*(e_for_compute+norm(e_for_compute,2));
while (sqrt((b-b_to_use)'*(b-b_to_use))>eta_to_use)&& (times_for_train<=times_for_train_max)
    
    
    b_to_use=b;
    
    
    %%-------------------------------------------------------------------------
    %get v
    %%-------------------------------------------------------------------------
    %--------------------------------------------------------------------------
    %--------------------------------------------------------------------------
    Iav=eye(d2+1);
    Iav(d2+1,d2+1)=0;
    %get Y
    Y=zeros(size(cluster_all_for_this_classifier,1)-1,d2+1);
    
    for row_Y=1:size(cluster_all_for_this_classifier,1)
        yn=cluster_all_for_this_classifier{row_Y,2}*[u_to_use'*cell2mat(cluster_all_for_this_classifier{row_Y,1}),1]';
        Y(row_Y,:)=yn';
    end
    %get v
    W_panduan_hanyou_nan_or_inf=isnan(c*Iav+Y'*Y);
    W_nan=0;
    for row=1:size(W_panduan_hanyou_nan_or_inf,1)
        for col=1:size(W_panduan_hanyou_nan_or_inf,2)
            W_nan=W_panduan_hanyou_nan_or_inf(row,col)+W_nan;
        end
    end
    W_panduan_hanyou_nan_or_inf=isinf(c*Iav+Y'*Y);
    W_inf=0;
    for row=1:size(W_panduan_hanyou_nan_or_inf,1)
        for col=1:size(W_panduan_hanyou_nan_or_inf,2)
            W_inf=W_panduan_hanyou_nan_or_inf(row,col)+W_inf;
        end
    end
    if W_nan>0||W_inf>0
        break;%denote Iav+lambda/2*(A_for_compute+A_for_compute')+c*Y'*Y cover NaN or Inf and stop training
    end
    if rank((c*Iav+Y'*Y))~=(d2+1)
        A_A=(c*Iav+Y'*Y)+eye(d2+1);
    elseif rank((c*Iav+Y'*Y))==(d2+1)
        A_A=(c*Iav+Y'*Y);
    end
    if rcond(A_A)<0.000000000000000000001
        break;
    end
    
    v=inv(A_A)*Y'*(ones(size(cluster_all_for_this_classifier,1),1)+b_to_use);
    v=v/norm(v,2);
    W_panduan_hanyou_nan_or_inf=isnan(v);
    W_nan=0;
    for row=1:size(W_panduan_hanyou_nan_or_inf,1)
        for col=1:size(W_panduan_hanyou_nan_or_inf,2)
            W_nan=W_panduan_hanyou_nan_or_inf(row,col)+W_nan;
        end
    end
    W_panduan_hanyou_nan_or_inf=isinf(v);
    W_inf=0;
    for row=1:size(W_panduan_hanyou_nan_or_inf,1)
        for col=1:size(W_panduan_hanyou_nan_or_inf,2)
            W_inf=W_panduan_hanyou_nan_or_inf(row,col)+W_inf;
        end
    end
    if W_nan>0||W_inf>0
        break;%denote v cover NaN or Inf, and stop training
    end
    v_to_use_1=v;
    v_to_use=v_to_use_1(1:size(v_to_use_1,1)-1,size(v_to_use_1,2));
    v0_to_use=v_to_use_1(size(v_to_use_1,1),size(v_to_use_1,2));
    %%-------------------------------------------------------------------------
    %get u
    %%-------------------------------------------------------------------------
    
    
    I=eye(d1);
    %compute¡°c\sum_{n=1}^NA_nv^{\sim}v^{\sim T}A_n^T¡±
    A_for_compute=zeros(d1,d1);%to store result of '\sum_{n=1}^NA_nv^{\sim}v^{\sim T}A_n^T'
    for row_for_sum_A=1:size(cluster_all_for_this_classifier,1)
        A=(cluster_all_for_this_classifier{row_for_sum_A,1});
        if iscell(A)==1
            A=cell2mat(A);
        end
        
        A_for_compute=A_for_compute+A*v_to_use*v_to_use'*A';
    end

    
    %compute [I+c\sum_{n=1}^NA_nv^{\sim}v^{\sim
    %T}A_n^T+\lambda\sum{i}\sum{j}(A_j-A_i^{\sim})v^{\sim}v^{\sim T}(A_j-A_i^{\sim})^T]^{-1}
    temp_left_of_u=(c*I+A_for_compute);
    if rank(temp_left_of_u)~=d1
        temp_left_of_u=temp_left_of_u+0.001*eye(d1);
    end
    left_of_u=inv(temp_left_of_u);
    %compute right_of_u, i.e., c\sum_{n=1}^N\varphi_nA_nv^{\sim}(1+b_n-\varphi_nv_0)
    
    B_for_compute=zeros(d1,1);%to store result of 'c\sum_{n=1}^N\varphi_nA_nv^{\sim}(1+b_n-\varphi_nv_0)'
    for row_for_sum_B=1:size(cluster_all_for_this_classifier,1)
        B=(cluster_all_for_this_classifier{row_for_sum_B,1});
        if iscell(B)==1
            B=cell2mat(B);
        end
        
        B_for_compute=B_for_compute+B*v_to_use*(1+b_to_use(row_for_sum_B)-cluster_all_for_this_classifier{row_for_sum_B,2}*v0_to_use);
    end
    right_of_u=B_for_compute;
    u=left_of_u*right_of_u;
    
    
    
    
    %normalized 
    
    u=u/norm(u,2);
    W_panduan_hanyou_nan_or_inf=isnan(u);
    W_nan=0;
    for row=1:size(W_panduan_hanyou_nan_or_inf,1)
        for col=1:size(W_panduan_hanyou_nan_or_inf,2)
            W_nan=W_panduan_hanyou_nan_or_inf(row,col)+W_nan;
        end
    end
    W_panduan_hanyou_nan_or_inf=isinf(u);
    W_inf=0;
    for row=1:size(W_panduan_hanyou_nan_or_inf,1)
        for col=1:size(W_panduan_hanyou_nan_or_inf,2)
            W_inf=W_panduan_hanyou_nan_or_inf(row,col)+W_inf;
        end
    end
    if W_nan>0||W_inf>0
        break;%denote u cover NaN or Inf and stop training
    end

    
    u_to_use=u;
    e_for_compute=(Y*v_to_use_1-ones(size(cluster_all_for_this_classifier,1),1)-b_to_use);
    
    %update b
    b=b_to_use+p*(e_for_compute+norm(e_for_compute,2));
    times_for_train=times_for_train+1;
    
end

u_border=u_to_use;
v_border=v_to_use;
v0=v0_to_use;









