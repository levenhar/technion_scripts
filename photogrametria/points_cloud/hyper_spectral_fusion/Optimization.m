clc
clear all
%% initial Vaule

PointsCloud = csvread("FullCloud.csv");

% PointsCloudN = normalize(PointsCloud(:,4:13),1);
PointsCloudNALL = {normalize(PointsCloud(:,4:13),1), normalize(PointsCloud(:,1:13),1) , normalize(PointsCloud,1),normalize(PointsCloud(:,4:end),1),normalize(PointsCloud,1),normalize(PointsCloud(:,14:end),1)};



%% Optimization 
k=7;
NoI=5;

q=1;
c=1;
for m = 1:length(PointsCloudNALL)
    PointsCloudN = PointsCloudNALL{m};
    for k = 4
        for n = 1:NoI
            try
                Centers0 = rand(k, size(PointsCloudN, 2));
                %set the optimization options
                options = optimoptions('fmincon','Display','off',"MaxFunctionEvaluations",30000);
                
                % define the varaible names and thier size
                centroids=optimvar("centroids",[k size(PointsCloudN, 2)]);
                
                % define the optimization problem
                convObj = fcn2optimexpr(@costFunction, PointsCloudN, k, centroids,'OutputSize', [1,1]);
                p=optimproblem("Description","findTab","Objective",convObj);
                p.ObjectiveSense="minimize";
                
                % generate an initial estimations. i.e., x0
                x0.centroids=Centers0;
        
                %x0 = optimvalues(p,'centroids',Centers0);
                
                % solve the optimization problem
                
                [xe,fval,exitflag,output]=solve(p,x0,'Options',options);
                
                FVAL{m}(n,k-3)=fval;
                XE{m}(n,k-3).xe=xe;
                q=q+1;
            catch
    %             FVAL(n,k-3)=nan;
                c=c+1;
            save("FVAL_4groups.mat", "FVAL", '-v7.3')
            save("XE_4groups.mat", "XE", '-v7.3')
            end
        end 
    end
end
save("FVAL_4groups.mat", "FVAL", '-v7.3')
save("XE_4groups.mat", "XE", '-v7.3')

[~, Oind] = find(FVAL==min(FVAL(:)));
xe=XE(Oind(1)).xe;
[~, indices1] = pdist2(xe.centroids, PointsCloudN, 'squaredeuclidean',"Smallest",1);


ClassefiedCloud = [PointsCloud(:,1:3) transpose(indices1)];

writematrix(ClassefiedCloud, 'ClassefiedCloud4Groups_JustSpec.csv');

