
clc;
clear all;
close all;

% path of converted mask
% /media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/DiscoGAN/F04/mask

arch = {'DiscoGAN', 'DNN'};
gen = {'F02','F03','F05','M04','M08','M09','M10','M12'};

for ar=1:length(arch)
    for gn=1:length(gen)

        % path of converted mask
        filelist = dir(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/',arch{ar},'/',gen{gn},'/mask/*.mat']);
        baseFileNames = natsortfiles({filelist.name});
        
        % path of existing mcc to match name of converted mcc
        filelist1 = dir(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/',gen{gn},'/dysarthric/testing_feat/mcc/*.mcc']);
        baseFileNames1 = natsortfiles({filelist1.name});
        
        foo = [];
        x = [];
        for i=1:length(filelist)  
            load(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/',arch{ar},'/',gen{gn},'/mask/',baseFileNames{i}]); 
            x = foo';
            [a,b] = size(x);
            x = reshape(x,a*b,1);
            disp(['Processing file : ', baseFileNames{i} , ' original:', filelist1(i).name])
            
            fid = fopen(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/',arch{ar},'/',gen{gn},'/converted_mcc/',filelist1(i).name],'w');
            fwrite(fid,x,'float');
            fclose('all');
        end
    end
end

