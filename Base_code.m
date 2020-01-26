%Base code
filedir1 = dir('D:/Neuronets/B_W_Ex1/Angry/*.wav');   %list the current folder content for .wav file 
fileRoot1 =  fileparts('D:/Neuronets/B_W_Ex1/Angry/*.wav');
Y1 = cell(1,length(filedir1));  %pre-allocate Y in memory (edit from @ Werner) 
FS1 = Y1;       %pre-allocate FS in memory (edit from @ Werner) 
for i = 1:length(filedir1)  %loop through the file names 
    %read the .wav file and store them in cell arrays 
    str = fullfile(fileRoot1,filedir1(i).name);
    [Y1{i}, FS1{i}] = audioread(str); 
end
for k = 1:length(Y1)
        startSample = FS1{1}*1.0;
        endSample = FS1{1}*3.12;
        Y_angry{1,k} = Y1{1,k}(startSample:endSample);
end
filedir2 = dir('D:/Neuronets/B_W_Ex1/Other/*.wav');   %list the current folder content for .wav file 
fileRoot2 =  fileparts('D:/Neuronets/B_W_Ex1/Other/*.wav');
Y2 = cell(1,length(filedir2));  %pre-allocate Y in memory (edit from @ Werner) 
FS = Y2;       %pre-allocate FS in memory (edit from @ Werner) 
for i = 1:length(filedir2)  %loop through the file names 
    %read the .wav file and store them in cell arrays 
    str2 = fullfile(fileRoot2,filedir2(i).name);
    [Y2{i}, FS2{i}] = audioread(str2); 
end
for k = 1:length(Y2)
        startSample = FS2{1}*1.0;
        endSample = FS2{1}*3.12;
        Y_other{1,k} = Y2{1,k}(startSample:endSample);
end
Y_other{1,1} = Y_other{1,1}';
X_input_an = cell2mat(Y_angry);
X_input_oth = cell2mat(Y_other);
len=1e3;
Nfix=fix(size(X_input_an,1)/len);
Input_data_an=reshape(X_input_an(1:Nfix*len,:),len,[]);
Input_data_oth=reshape(X_input_oth(1:Nfix*len,:),len,[]);
% Input_data_an = [X_input_an(1:1e3+1,:) X_input_an(1e3:2e3,:) X_input_an(2e3:3e3,:) X_input_an(3e3:4e3,:) X_input_an(4e3:5e3,:) X_input_an(5e3:6e3,:) X_input_an(6e3:7e3,:) X_input_an(7e3:8e3,:) X_input_an(8e3:9e3,:) X_input_an(9e3:10e3,:)];
% Input_data_oth = [X_input_oth(1:1e3+1,:) X_input_oth(1e3:2e3,:) X_input_oth(2e3:3e3,:)  X_input_oth(3e3:4e3,:) X_input_oth(4e3:5e3,:) X_input_oth(5e3:6e3,:) X_input_oth(6e3:7e3,:) X_input_oth(7e3:8e3,:) X_input_oth(8e3:9e3,:) X_input_oth(9e3:10e3,:)];
Input_data0 = [Input_data_an Input_data_oth];
xmin=repmat(min(Input_data0),size(Input_data0,1),1);
Input_data1=Input_data0-xmin;
xmax=repmat(max(Input_data1),size(Input_data1,1),1);
Input_data=Input_data1./xmax;
ind_an=zeros(2,size(Input_data_an,2));
ind_an(1,:)=1;
ind_oth=zeros(2,size(Input_data_oth,2));
ind_oth(2,:)=1;
% zero = zeros(1,((length(Input_data(1,:)))/2));
% one = ones(1,((length(Input_data(1,:)))/2));
T = [ind_an ind_oth];
hiddenSize = 10;
autoenc1 = trainAutoencoder(Input_data,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.05,...
    'DecoderTransferFunction','purelin'); %350 epochs
features1 = encode (autoenc1, Input_data);
autoenc2 = trainAutoencoder(features1,hiddenSize,...
   'L2WeightRegularization',0.001,...
   'SparsityRegularization',4,...
   'SparsityProportion',0.05,...
   'DecoderTransferFunction','purelin',...
   'ScaleData',false);
features2 = encode(autoenc2,features1);
softnet = trainSoftmaxLayer (features2,T, 'LossFunction' , 'crossentropy' );
deepnet_1 = stack(autoenc1, autoenc2, softnet);
deepnet_1 = train(deepnet_1,Input_data,T);
sign_type = deepnet_1(Input_data);
plotconfusion(T, sign_type);
save ('deepnet_1');
