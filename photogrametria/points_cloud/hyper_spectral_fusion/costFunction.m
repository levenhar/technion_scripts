function [value] = costFunction(Fetures,k,estimatevalue)

% Find the closest center for each point
[~, indices] = pdist2(estimatevalue, Fetures, "squaredeuclidean","Smallest",1);

% Ls=unique(indices);

% Create a cell array to hold the k arrays of points
%arrays = cell(k,1);

Lf=length(Fetures);
SumS=0;
for i=1:k
    F=Fetures(indices==i,:);
    C(i,:)=mean(F);
    %C1(i,:)=mean(F(:,1:3));
    t=abs(F-C(i,:));
    S(i,:)=mean(t(:));
    P(i)=length(F)/Lf;
    Si(i,:)=sum(S(i,:));
    SumS=SumS+sum(S(i,:));
end

Distances = pdist2(C,C,'squaredeuclidean',"Smallest",2);
invSumD = 1/sum(Distances(:));


% diffPortion = abs(portionVector'-P);
% minV=-inf;
% while minV<inf
%     minV=min(diffPortion(:));
%     [r, c] = find(diffPortion==minV);
%     C2(r(1),:)= C(c(1),:);
%     
%     P2(r(1))= P(c(1));
%     diffPortion(r(1),:) = inf;
%     diffPortion(:,c(1)) = inf;
% end
%------------------------------------------------------------

% perms_list1 = perms(P);
% diffs = diag((perms_list1 - P)*(perms_list1 - P)');
% [~, min_idx] = min(diffs);
% min_permutation = perms_list1(min_idx, :);
% [~, idx_order] = ismember(min_permutation, P);
% 
% C=C(idx_order,:);
% P=P(:,idx_order);

%------------------------------------------------------------

% C=C2;
%P=P2;


% Si = Si./v';
% SumS = sum(Si);


diff=abs(C-estimatevalue);
diff=sum(diff,2);
Ndiff = sum(diff(:));
% Ndiff=norm(diff,"fro");


value=invSumD+SumS+Ndiff;

end