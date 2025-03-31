wavfiles = dir(fullfile('G:\My Drive\Studenten\Maike\Stickleback_templates_6045-audio','*.wav'));
filt_low = 20;
filt_high = 500;
fs = 24000;

s1 = audioread([wavfiles(1).folder,'\',wavfiles(1).name]);
[b,a] = ellip(4,0.1,40,[filt_low,filt_high]*2/fs);
    s1_filt = filter(b,a,s1);

p1 = find(s1 == max(s1));

s2 = audioread([wavfiles(2).folder,'\',wavfiles(2).name]);
p2 = find(s2 == max(s2));

d1 = p1(1) - p2(1);

figure;plot(s1)
figure;plot(s2)

allsounds = zeros(length(s1),length(wavfiles));
allsounds(:,1) = s1;
allsounds(d1+1:end,2) = s2(1:end-d1);

figure;plot(allsounds(:,1))
hold on
plot(allsounds(:,2))

for n = 2:length(wavfiles)
    sd = audioread([wavfiles(n).folder,'\',wavfiles(n).name]);
    [b,a] = ellip(4,0.1,40,[filt_low,filt_high]*2/fs);
    sd_filt = filter(b,a,sd);

    pd = find(sd_filt == max(sd_filt));

    dn = p1(1) - pd(1);

    if dn>0
        allsounds(dn+1:end,n) = sd_filt(1:end-dn);
    else
        allsounds(1:end-abs(dn),n) = sd_filt(abs(dn)+1:end);
    end
end

figure;plot(allsounds(:,1))
hold on
plot(allsounds(:,8))
hold off

meansound = mean(allsounds,2);
mediansound = median(allsounds,2);
figure;plot(meansound)

save('C:\Users\P310512\Documents\GitHub\Triton\Remoras\Fish detector\Stickleback_mean_template.mat','meansound')