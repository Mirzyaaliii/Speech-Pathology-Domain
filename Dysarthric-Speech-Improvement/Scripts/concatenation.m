
%%% Saving Features with and without Outliers for different configurations
clc; 
clear all; 
close all;


gen = {'F04', 'M05'};
SRC = {'dysarthric'};
TGT = {'control'};
tr_vl = {'training_', 'validation_'};

% path of source speaker
% /media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/F04/dysarthric/training_feat/mcc
% path of target speaker
% /media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/F04/control/training_feat/mcc



dim=40;result=[];mpsrc=[];mptgt=[];t_scores=[];l=[];lz=[];wr=[];mpsrc1=[];mptgt1=[];Z1=[];Z=[];Z2=[];
x=[];y=[];X=[];Y=[];path=[];

filelist = dir(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/',gen{2},'/',SRC{1},'/',tr_vl{1},'feat/mcc/*.mcc']);


for index=1:length(filelist)

    fprintf('Processing %s\n',filelist(index).name);

    fid=fopen(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/',gen{2},'/',SRC{1},'/',tr_vl{1},'feat/mcc/',filelist(index).name]);
    x=fread(fid,Inf,'float');
    x=reshape(x,dim,length(x)/dim);

    fid1=fopen(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/',gen{2},'/',TGT{1},'/',tr_vl{1},'feat/mcc/',filelist(index).name]);
    y=fread(fid1,Inf,'float');
    y=reshape(y,dim,length(y)/dim);         

    % align features of source and target speaker
    [min_distance, d, g, path] = dtw_E(x, y);       

    X = [X x(:,path(:,1))];
    Y = [Y y(:,path(:,2))];

    fclose('all');
end

% concatenation of aligned features in Z.mat 
Z = [X;Y];

save(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/',gen{2},'/Z.mat'],'Z');    




