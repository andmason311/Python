clear all
clc
close all
% Read in Hardness Data and Assign to Array Variables
cd C:\Users\Admin\OneDrive\Documents\School\AFIT_Classes\Quarter_4\NENG_612\Lab_3\Data
Data = readtable('Vickers_Hardness.xlsx');
BR=table2array(Data(:,4)); BL=table2array(Data(:,3)); TR=table2array(Data(:,2)); TL=table2array(Data(:,1));
% Remove NaN Values
BR = BR(~isnan(BR))'; BL = BL(~isnan(BL))'; TR = TR(~isnan(TR))'; TL = TL(~isnan(TL))';
%Calculate Average and Standard Error for Each Array
Avg=[mean(TL), mean(TR), mean(BL), mean(BR)];
StandardError=[std(TL)/sqrt(length(TL)),std(TR)/sqrt(length(TR)),std(TL)/sqrt(length(TL)),std(TL)/sqrt(length(TL))];
%Set Random Array xaxis and point colors
Sample=[.5,1,1.5,2];
c = [0 1 0; 1 0 0; 0.5 0.5 0.5; 0.6 0 1];
%For Loop to Make Each Point in the Legend
figure;
hold on
for k=1:numel(Sample)
scatter(Sample,Avg,700,c,'s','filled');
end
hold on;
%Overlay Error Bars
errorbar(Sample,Avg,StandardError, 'LineStyle','none');
%Assign Cosmetic Graph Attributes
axis([0,2.5,200,260]);
ylabel('Vickers Hardness(HV)','FontSize',30); title('Vickers Hardness with Stardard Error','FontSize',40);
set(gca,'XTick',[])
legend({'Back Left', 'Back Right', 'Front Left', 'Front Right', 'Error'},'FontSize',30);