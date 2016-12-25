function [Mat_Test]=matrixlize_for_test(dataset_for_normalize,field_num,d1,d2)

testsample=zeros(1,field_num+1);
jjj=1;
for test1=1:size(dataset_for_normalize,2)
    for numtest=1:size(dataset_for_normalize(1,test1),1)
        for order_test_this_class=1:size(dataset_for_normalize(1,test1).class,1)
        testsample(jjj,:)=dataset_for_normalize(1,test1).class(order_test_this_class,:);
        jjj=jjj+1;
        end
    end
end
testsample_mat=cell(size(testsample,1)+1,3);
testsample_mat{1,1}='matrixzied result of sample';
testsample_mat{1,2}='real class label of sample';
testsample_mat{1,3}='test class label of sample';
for ru=1:size(testsample,1)
    temp=testsample(ru,1:size(testsample,2)-1);
    mat_testsample=reshape(temp,d1,d2);
    testsample_mat{1+ru,1}=mat_testsample;
    testsample_mat{1+ru,2}=testsample(ru,size(testsample,2));
end
Mat_Test=testsample_mat;