function main_for_EMatMHKS
load('data.mat');
%select pima data set
result_for_last2=juleijieguo;
sub_class_trainA=result_for_last2(4,:);%Each time, please use only one data set¡£
dataset_name=sub_class_trainA(1,1);
training_set=sub_class_trainA(1,2);
test_set=sub_class_trainA(1,3);
%set up min-to-maj class ratio
ratio_mir_to_maj=35/65;
positive_class_label=0;%set positive-class label
%parameters 
u_for_use=0.1:0.1:1;
v_for_use=0.1:0.1:1;
p_for_use=0.99;
c_for_use=[0.001,0.01,0.1,1,10,100,1000];
lambda=1;%this parameter has no sense
v0_for_use=1;
eta_for_use=10^-4;
b_for_use=10^-6;
k_neighbor=10;%please note, we can change the parameters for different cases
%-------------------------------------------------------------------------
%Initialize the table of results
%Initialize result_for_last1 which is used to store the final results
result_for_last1=cell(1,21);
result_for_last1{1,1}='data name';
result_for_last1{1,2}='clustering method,0--by sample,1--by distance';
result_for_last1{1,3}='clsutering way, from 0 to 1';
result_for_last1{1,4}='no. attributes';
result_for_last1{1,5}='no. class';
result_for_last1{1,6}='method name';
result_for_last1{1,7}='';
result_for_last1{1,8}='c';
result_for_last1{1,9}='initial value of u';
result_for_last1{1,10}='initial value of v';
result_for_last1{1,11}='initial value of b';
result_for_last1{1,12}='p';
result_for_last1{1,13}='lambda';
result_for_last1{1,14}='d1';
result_for_last1{1,15}='d2';
result_for_last1{1,16}='no. error test sample';
result_for_last1{1,17}='test recognition';
result_for_last1{1,18}='time complex';
result_for_last1{1,19}='space complex';
result_for_last1{1,20}='info of classifier';
result_for_last1{1,21}='info of clusters';
result_for_last1{1,22}='no. iteration';
%%get the fuzzy relationship for each sample
training_num=0;
test_num=0;
%get the total number of training and test samples
for tr_num_row=1:size(training_set{1,1},2)
    training_num=training_num+size(training_set{1,1}(1,tr_num_row).class,1);
end%ÑµÁ·ÊýÄ¿
for te_num_row=1:size(test_set{1,1},2)
    test_num=test_num+size(test_set{1,1}(1,te_num_row).class,1);
end%no. training
class_num=size(training_set{1,1},2);%no. class
training_set_t=training_set{1,1};
test_set_t=test_set{1,1};
fuzzy_membership=zeros(training_num,2);%first column of fuzzy_membership: class of sample;second column of fuzzy_membership: fuzzy extent of sample
%set up a matrix to store the no. training samples and their class labels for each class
train_set_num_of_class=zeros(class_num,2);
for row_of_train_set_num_of_class=1:1:class_num
    train_set_num_of_class(row_of_train_set_num_of_class,1)=training_set_t(row_of_train_set_num_of_class).class(1,size(training_set_t(row_of_train_set_num_of_class).class,2));
    train_set_num_of_class(row_of_train_set_num_of_class,2)=size(training_set_t(row_of_train_set_num_of_class).class,1);
end
%%put the training samples in together
training_all=zeros(training_num,size(training_set_t(1).class,2));
for row_of_train_set_num_of_class=1:1:class_num
    if row_of_train_set_num_of_class==1
        training_all(1:train_set_num_of_class(1,2),:)=training_set_t(1).class;
    else
        total_present=0;
        for total_present_row=1:1:row_of_train_set_num_of_class-1
            total_present=total_present+train_set_num_of_class(total_present_row,2);
        end
        training_all(total_present+1:total_present+train_set_num_of_class(row_of_train_set_num_of_class,2),:)=training_set_t(row_of_train_set_num_of_class).class;
    end
end
for row_of_pattern=1:1:training_num
    present_pattern=training_all(row_of_pattern,1:size(training_all,2)-1);
    %compute the distance between present sample and others
    distance_matrix=zeros(training_num-1,2);%first column:store distance, second column:store class label
    row_of_distance_matrix=1;
    for row_for_distance_matrix=1:1:training_num
        if row_for_distance_matrix==row_of_pattern
        else
            distance_matrix(row_of_distance_matrix,1)=norm(present_pattern-training_all(row_for_distance_matrix,1:size(training_all,2)-1),2);
            distance_matrix(row_of_distance_matrix,2)=training_all(row_for_distance_matrix,size(training_all,2));
            row_of_distance_matrix=row_of_distance_matrix+1;
        end
    end
    %sort the distance_matrix
    distance_matrix=sortrows(distance_matrix,1);
    %make sure the no. positive samples and negative samples for the top k samples in distance_matrix
    positive_class=positive_class_label;
    num_positive=0;
    num_negetive=0;
    for row_of_distance_matrix_sort=1:1:k_neighbor
        if positive_class==distance_matrix(row_of_distance_matrix_sort,2);
            num_positive=num_positive+1;
        else
            num_negetive=num_negetive+1;
        end
    end
    p_positive=num_positive/k_neighbor;
    p_negetive=num_negetive/k_neighbor;
    %compute the entropy
    if p_positive==0&&p_negetive==0
        H_x=0;
    elseif p_positive==0&&p_negetive~=0
        H_x=-p_positive*log(p_positive);
    elseif p_positive~=0&&p_negetive==0    
        H_x=-p_negetive*log(p_negetive);
    else
        H_x=-p_negetive*log(p_negetive)-p_positive*log(p_positive);
    end
    %compute fuzzy_member
    if training_all(row_of_pattern,size(training_all,2))==positive_class_label
        fuzzy_member=1-H_x;
    else
        fuzzy_member=(1-H_x)*ratio_mir_to_maj;
    end
    %store fuzzy_member
    fuzzy_membership(row_of_pattern,1)=training_all(row_of_pattern,size(training_all,2));
    if isnan(fuzzy_member)==1
    else
    fuzzy_membership(row_of_pattern,2)=fuzzy_member;
    end
end
%update the training set itself, i.e., multiplied by the fuzzy_member
sub_class_train=sub_class_trainA{1,2};
%counter, row_of_pattern
row_of_pattern=1;
for row_of_class=1:1:size(sub_class_train,2)
    temp=sub_class_train(row_of_class).class;%record the data temporary
    for row_of_temp=1:1:size(temp,1)
        temp(row_of_temp,1:size(temp,2)-1)=temp(row_of_temp,1:size(temp,2)-1)*fuzzy_membership(row_of_pattern,2);
        row_of_pattern=row_of_pattern+1;
    end
    sub_class_train(row_of_class).class=temp;
end
sub_class_trainA{1,2}=sub_class_train;
for u_for_use_order=1:size(u_for_use,2)
    for v_for_use_order=1:size(v_for_use,2)
        for p_for_use_order=1:size(p_for_use,2) 
            for c_for_use_order=1:size(c_for_use,2)
                for v0_for_use_order=1:size(v0_for_use,2)
                    for b_for_use_order=1:size(b_for_use,2)
                        for eta_for_use_order=1:size(eta_for_use,2)
                            for k_sub_class_train=1:size(sub_class_trainA,1)
                                name=sub_class_trainA{k_sub_class_train,1};
                                test=sub_class_trainA{k_sub_class_train,3};
                                sub_class_train=sub_class_trainA{k_sub_class_train,2};
                                u=u_for_use(1,u_for_use_order);
                                v=v_for_use(1,v_for_use_order);
                                p=p_for_use(1,p_for_use_order);
                                c=c_for_use(1,c_for_use_order);
                                v0=v0_for_use(1,v0_for_use_order);
                                b=b_for_use(1,b_for_use_order);
                                eta=eta_for_use(1,eta_for_use_order);
                                [result_for_last1,row]=EMatMHKS(result_for_last1,sub_class_train,test,u,v,p,c,lambda,b,v0,eta);
                                result_for_last1{row-1,1}=name;
                                result_for_last1{row-1,6}='EMatMHKS';
                            end
                        end
                    end
                end
            end
        end
    end
end