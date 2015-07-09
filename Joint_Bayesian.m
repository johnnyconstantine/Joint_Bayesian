%---------------------Load the training data------------------------
load('lbp_WDRef.mat')%there are 71846 pictures,each contains 5900 £¨71846*5900£©
load('id_WDRef.mat')%persion id ,same id means same person
labels = id_WDRef;
clear id_WDRef;
X = double(lbp_WDRef);%trans int into double
clear lbp_WDRef;
train_mean = mean(X,1);
X = bsxfun(@minus,X,train_mean);%substract the mean
[COEFF,SCORE] = princomp(X,'econ');%PCA
train_x = SCORE(:,1:2000);%
%----------------------------------------------------------
clear X;
clear SCORE;
%----------------------------------------------------------
%----------------------------------------------------------
%------------------------------load testing data-------------------------------------
load('lbp_lfw.mat')%it's 13233*5900
load('pairlist_lfw.mat')
%pairlist_lfw is an struct£¬it contains two 3000*2 matrix£¬they are IntraPersonPair¡¢ExtraPersonPair
normX = double(lbp_lfw);
train_mean = mean(normX,1);
normX = bsxfun(@minus,normX,train_mean);
normX = normX * COEFF(:,1:2000);%PCA
test_Intra = pairlist_lfw.IntraPersonPair;
test_Extra = pairlist_lfw.ExtraPersonPair;
%----------------------------------------------------------------------------------------------------------
clear lbp_lfw;
clear pairlist_lfw;
clear train_mean;
clear COEFF;
%-------------------------creat training pairs£¨intraPairºÍextraPair£©---------------------------------------------------
[classes, bar, labels] = unique(labels);%labels
    nc = length(classes);%there are 2995 person
    clear bar and classes;
train_Intra = zeros(nc*2,2);%5990*2
for i=1:nc%randperm£¨n£¬2£©
    train_Intra(2*i-1,:) = randperm(sum(labels == i),2) + find(labels == i,1,'first') - 1;
    train_Intra(2*i,:) = randperm(sum(labels == i),2) + find(labels == i,1,'first') - 1;
end;%choose two pictures from one person as intraPairs
train_Extra = reshape(randperm(length(labels),20000),10000,2);%reshape 10000*2
train_Extra(labels(train_Extra(:,1))==labels(train_Extra(:,2)),:)=[];
train_Extra(size(train_Intra,1)+1:end,:)=[];
Dis_train_Intra = zeros(size(train_Intra,1),1);
Dis_train_Extra = zeros(size(train_Intra,1),1);
%------------------------------initialize before training---------------------------------------------------------
    m = length(labels); % 71846
    n = size(train_x,2);%
	[classes, bar, labels] = unique(labels);
    nc = length(classes);%Îª2995
    clear classes;
    clear bar;
	Sw = eye(size(train_x, 2), size(train_x, 2));
    Su = eye(size(train_x, 2), size(train_x, 2));%initialize sw su as identity matrix
    cur = {};%cell£¬0*0
    withinCount = 0;
    numberBuff = zeros(1000,1);% make sure that the largest number of a person is less than 1000
    for i=1:nc%
        cur{i} = train_x(labels == i,:);%make cell
        if size(cur{i},1)>1
            withinCount = withinCount + size(cur{i},1);
        end;
        if numberBuff(size(cur{i},1)) == 0
            numberBuff(size(cur{i},1)) = 1;
        end;
    end;%
    clear labels;
    u = zeros(n,nc);%2000*2995
    clear withinCount;
    oldSw = Sw;
    SuFG = cell(1000,1);
    SwG = cell(1000,1);%1000*1 cell
    %-------------------------------------EM-Like begin training-------------------------------------------------
    for k=1:150 %set the cycle times
        F = inv(Sw);%formula£¨5£©in supplementary material
        ep =zeros(n,m);
        nowp = 1;
        for g = 1:1000
            if numberBuff(g)==1
                G = -1 .* (g .* Su + Sw) \ Su / Sw;%formula£¨6£©in supplementary material
                SuFG{g} = Su * (F + g.*G);%
                SwG{g} = Sw*G;
            end;
        end;%1000
        for i=1:nc%
            nnc = size(cur{i}, 1);%number of each person
            u(:,i) = sum(SuFG{nnc} * cur{i}',2);%
            ep(:,nowp:nowp+ size(cur{i}, 1)-1) = bsxfun(@plus,cur{i}',sum(SwG{nnc}*cur{i}',2));%formula£¨8£©in supplementary material
            nowp = nowp+ nnc;
        end;
        Su = cov(u');%su£¬
        Sw = cov(ep');%sw£¬¡£
        fprintf('%d %f\n',k,norm(Sw - oldSw)/norm(Sw));
        if norm(Sw - oldSw)/norm(Sw)<1e-6%
            break;
        end;
        oldSw = Sw;       
    F = inv(Sw);
    mapping.G = -1 .* (2 * Su + Sw) \ Su / Sw;%formula£¨6£©in supplementary material
    mapping.A = inv(Su + Sw) - (F + mapping.G);%formula£¨5£©
    mapping.Sw = Sw;
    mapping.Su = Su;
%---------------------------------begin testing-------------------------------------------------------------------

    for i=1:size(train_Intra,1)%formula£¨4£©
        Dis_train_Intra(i) = train_x(train_Intra(i,1),:) * mapping.A * train_x(train_Intra(i,1),:)' + train_x(train_Intra(i,2),:) * mapping.A * train_x(train_Intra(i,2),:)' - 2 * train_x(train_Intra(i,1),:) * mapping.G * train_x(train_Intra(i,2),:)';
        Dis_train_Extra(i) = train_x(train_Extra(i,1),:) * mapping.A * train_x(train_Extra(i,1),:)' + train_x(train_Extra(i,2),:) * mapping.A * train_x(train_Extra(i,2),:)' - 2 * train_x(train_Extra(i,1),:) * mapping.G * train_x(train_Extra(i,2),:)';
    end;
    group_train = [ones(size(Dis_train_Intra,1),1);zeros(size(Dis_train_Extra,1),1)];%
    training = [Dis_train_Intra;Dis_train_Extra];%11980*1
%-------------------------------------------------------------------------------------------------------------
    result_Intra = zeros(3000,1);
    result_Extra = zeros(3000,1);
    for i=1:3000
        result_Intra(i) = normX(test_Intra(i,1),:) * mapping.A * normX(test_Intra(i,1),:)' + normX(test_Intra(i,2),:) * mapping.A * normX(test_Intra(i,2),:)' - 2 * normX(test_Intra(i,1),:) * mapping.G * normX(test_Intra(i,2),:)';
        result_Extra(i) = normX(test_Extra(i,1),:) * mapping.A * normX(test_Extra(i,1),:)' + normX(test_Extra(i,2),:) * mapping.A * normX(test_Extra(i,2),:)' - 2 * normX(test_Extra(i,1),:) * mapping.G * normX(test_Extra(i,2),:)';
    end;%formula£¨4£©
    group_sample = [ones(3000,1);zeros(3000,1)];%
    sample = [result_Intra;result_Extra];%
%---------------------------------------
   % clear normX;%
%---------------------------------------
% classification  
    [p,q]=size(group_train);%11980*1
    starderd=0;
    eg1=0;
    num1=0;
    eg2=0;
    num2=0;
    for i=1:p
        if group_train(i,1)==0
            eg2=eg2+training(i,1);%training
            num2=num2+1;
        else
            eg1=eg1+training(i,1);
            num1=num1+1;
        end
    end
    starderd=(eg1+eg2)/(num1+num2);
    [p,q]=size(sample);%sample
    label=zeros(p,1);%
    accuracy=0;
    for i=1:p
        if sample(i,1)>starderd%same person
            label(i,1)=1;
        else
            label(i,1)=0;%not same person
        end
        if label(i,1)==group_sample(i,1)%
            accuracy=accuracy+1;
        end
    end
    result = accuracy/p %
    if(result>=0.90)%if you can't reach this high accuracy you may end this by ctl + c or just set a lower number
        clear Sw and Su and F and G and numberBuff;%this should be done by the end of the for cycle
        break;
    end;
 end;%new added by cx match the big for cycle
display('program done')
