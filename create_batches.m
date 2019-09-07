
clear all;
clc;
close all;

gen = {'F04', 'M05'};
tr_vl = {'training','validation'};

% load z.mat
b = fullfile(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/',gen{2},'/Z.mat']);
z = load(b);

a = (z.Z);
a = a';
[i, j] = size(a);


% number of batches rem/rem+1
rem = mod(i, 1000);
n = (i - rem)/1000;
disp(n)

temp = 0;
m = 1;


% create batch of size 1000X40
for k=1:n
    Feat = a(m:m+999, 1:40);
    Clean_cent = a(m:m+999, 41:80);
    
    %fprintf('m = %i',m)
    %fprintf('   temp = %i\n',temp)

    save(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/',gen{2},'/training_batches/Batch_',num2str(temp),'.mat'],'Feat','Clean_cent');    
    fprintf('Batch_%i created\n',temp);
        
    m = m + 1000;
    temp = temp + 1;
end

% if more than 700 rows are not containing zeros 
if rem>700
    b = zeros((1000 - rem), 40);
	
	Feat = [a(m:m+rem-1, 1:40); b];
    Clean_cent = [a(m:m+rem-1, 41:80); b];

    k = k + 1;
 
    save(['/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/',gen{2},'/training_batches/Batch_',num2str(temp),'.mat'],'Feat','Clean_cent');
    fprintf('Batch_%i created\n',temp);
    
end
