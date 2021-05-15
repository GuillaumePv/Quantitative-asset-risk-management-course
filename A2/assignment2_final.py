#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:14:53 2021

@author: sebastiengorgoni
"""

import pandas as pd
import numpy as np
import os 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from numpy.matlib import repmat

sns.set(style="whitegrid")

#os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Quantitative Asset & Risk Management/Assignments/Assignment 2")
print("Current working directory: {0}".format(os.getcwd()))

if not os.path.isdir('Plot'):
    os.makedirs('Plot')
    
if not os.path.isdir('Output'):
    os.makedirs('Output')

df_price = pd.read_excel('Data_QAM2.xlsx', 'Prices', engine='openpyxl')
df_price.set_index('Dates', inplace=True)

benchmark = pd.DataFrame(data=(0.5*df_price['World Equities'] + 0.5*df_price['World Bonds']))

in_sample_price = df_price.loc[(df_price.index <= pd.to_datetime('2010-12-31'))].iloc[:,:]
out_sample_price = df_price.loc[(df_price.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

returns_price = np.log(df_price/df_price.shift(1)).replace(np.nan, 0)
returns_IS_price = np.log(in_sample_price/in_sample_price.shift(1)).replace(np.nan, 0)

plt.plot((returns_price + 1).cumprod()*100)
plt.legend(returns_price.columns, loc='upper left', frameon=False)
plt.title('Cumulative Returns All Assets')
plt.savefig('Plot/returns_asset.png')
plt.show()
plt.close()

returns_OS_price = returns_price.loc[(returns_price.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

benchmark_return_IS = pd.DataFrame({'IS Benchmark': 0.5*returns_IS_price['World Equities'] + 0.5*returns_IS_price['World Bonds']})
benchmark_return_OS = pd.DataFrame({'OS Benchmark': 0.5*returns_OS_price['World Equities'] + 0.5*returns_OS_price['World Bonds']})

# =============================================================================
# =============================================================================
# #Part 1: SAA
# =============================================================================
# =============================================================================

# =============================================================================
# 1.1
# =============================================================================

def MCR_calc(alloc,Returns):
    ptf=np.multiply(Returns,alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    Sigma=np.cov(np.transpose(Returns))
    MCR=np.matmul(Sigma,np.transpose(alloc))/vol_ptf;
    return MCR

###ERC Allocation###
def ERC(alloc,Returns):
    ptf=np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    indiv_ERC=alloc*MCR_calc(alloc,Returns);
    criterion=np.power(indiv_ERC-vol_ptf/len(alloc),2)
    criterion=np.sum(criterion)*1000000000
    return criterion

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

cons=({'type':'eq', 'fun': lambda x:sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]-0.01},
      {'type':'ineq', 'fun': lambda x: x[2]-0.01})
Bounds= [(0 , 1) for i in range(0,7)]

res_ERC = minimize(ERC, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

labels=list(returns_IS_price)

plt.figure(figsize=(15,7))

#Plot the optimal weights using ERC
weight_to_chart=np.array(res_ERC.x)
plt.subplot(131)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation ERC')
plt.ylabel('Ptf Weight')
plt.tight_layout()

#MCR for Max SR
MCR_chart=MCR_calc(res_ERC.x, returns_IS_price)
MCR_chart=np.array(MCR_chart)
plt.subplot(132)
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR of ERC')
plt.ylabel('MCR')
plt.tight_layout()

#Test for the MCR/Risk ratio
ratio=res_ERC.x*MCR_chart
plt.subplot(133)
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('Alpha x MCR')
plt.ylabel('Ratio')
plt.tight_layout()
plt.savefig('Plot/ERC_output.png')
plt.show()
plt.close()

###Sharp Ratio Allocation###
def Max_SR(alloc, Returns, Rf=0):
    ptf=np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    mu_bar=np.mean(ptfReturns)-Rf;
    vol_ptf=np.std(ptfReturns);
    SR=-mu_bar/vol_ptf;
    return SR

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

cons=({'type':'eq', 'fun': lambda x:sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]-0.01},
      {'type':'ineq', 'fun': lambda x: x[2]-0.01})
Bounds= [(0 , 1) for i in range(0,7)]
#Bounds = [(0 , 1), (0.01 , 1), (0.01 , 1), (0 , 1), (0 , 1), (0 , 1), (0 , 1)]

res_SR = minimize(Max_SR, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

plt.figure(figsize=(15,7))

#Plot the optimal weights using Sharpe Ratio
weight_to_chart=np.array(res_SR.x)
plt.subplot(131)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation SR')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_SR.x, returns_IS_price)
MCR_chart=np.array(MCR_chart)
plt.subplot(132)
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR of SR')
plt.ylabel('MCR')
plt.tight_layout()

# Test for the Risk/Reward ratio
reward=np.mean(returns_IS_price, 0)
ratio=MCR_chart/reward
plt.subplot(133)
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('Excess Return to MCR Ratio')
plt.ylabel('Ratio')
plt.tight_layout()
plt.savefig('Plot/SR_output.png')
plt.show()
plt.close()

###Most-Diversified Portfolio###
def Max_DP(alloc,Returns):
    ptf=np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    
    numerator=np.multiply(np.std(Returns),alloc);
    numerator=np.sum(numerator);
    
    Div_Ratio=-numerator/vol_ptf;
    return Div_Ratio

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

cons=({'type':'eq', 'fun': lambda x: sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]-0.01},
      {'type':'ineq', 'fun': lambda x: x[2]-0.01})
Bounds= [(0 , 1) for i in range(0,7)]
#Bounds = [(0 , 1), (0.01 , 1), (0.01 , 1), (0 , 1), (0 , 1), (0 , 1), (0 , 1)]

res_MDP = minimize(Max_DP, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True}) #options={'disp': True} means I want feedbacks in my optimization

plt.figure(figsize=(15,7))

weight_to_chart=np.array(res_MDP.x)
plt.subplot(131)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation MDP')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_MDP.x, returns_IS_price);
MCR_chart=np.array(MCR_chart)
plt.subplot(132)
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR of MDP')
plt.ylabel('MCR')
plt.tight_layout()

#Test for MCR/Risk ratio
MCR_risk = MCR_chart/np.std(returns_IS_price)
plt.subplot(133)
plt.plot(labels, MCR_risk, 'ro', labels, MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR-per-Risk Ratio')
plt.ylabel('Ratio')
plt.tight_layout()
plt.savefig('Plot/MDP_output.png')
plt.show()
plt.close()

###Determine the performances of the IS  optimal portfolio allocatio###
portfolio_IS_ERC = pd.DataFrame({'ERC': np.sum(np.multiply(returns_IS_price,np.transpose(res_ERC.x)), 1)})
portfolio_IS_SR = pd.DataFrame({'SR': np.sum(np.multiply(returns_IS_price,np.transpose(res_SR.x)), 1)})
portfolio_IS_MDP = pd.DataFrame({'MDP': np.sum(np.multiply(returns_IS_price,np.transpose(res_MDP.x)), 1)})

def hit_ratio(return_dataset):
    return len(return_dataset[return_dataset >= 0]) / len(return_dataset)

def max_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    monthly_drawdown = cum_returns/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    return max_monthly_drawdown
    
max_drawdown((portfolio_IS_ERC['ERC']+1).cumprod()).min()

def perf(data, benchmark, name, name_plt):
    plt.figure(figsize=(20,7))
    plt.subplot(131)
    plt.plot(data, 'b')
    plt.title("Monthly Returns", fontsize=15)
    exp = np.mean(data,0)*12
    vol = np.std(data,0)*np.power(12,0.5)
    sharpe = exp/vol
    max_dd = max_drawdown((data+1).cumprod())
    plt.subplot(132)
    plt.plot(max_dd, 'g')
    plt.title("Evolution of Max Drawdown", fontsize=15)
    hit = hit_ratio(data)
    df = pd.DataFrame({name: [exp, vol, sharpe, max_dd.min(), hit]}, index = ['Mean', 'Volatility', 'SharpeRatio', 'MaxDrawdown', 'HitRatio'])
    plt.subplot(133)
    plt.plot((data + 1).cumprod()*100, 'b', label='Portfolio')
    plt.plot((benchmark + 1).cumprod()*100, 'r', label='Benchmark')
    plt.title(name_plt, fontsize=15)
    plt.legend(loc='upper left', frameon=True)
    plt.savefig('Plot/'+name_plt+'.png')
    plt.show()
    plt.close()
    return df

ERC_IS_results = perf(portfolio_IS_ERC['ERC'], benchmark_return_IS['IS Benchmark'], 'ERC', 'ERC Cumulative Returns In-Sample')
SR_IS_results = perf(portfolio_IS_SR['SR'], benchmark_return_IS['IS Benchmark'], 'SR', 'SR Cumulative Returns In-Sample')
MDP_IS_results = perf(portfolio_IS_MDP['MDP'], benchmark_return_IS['IS Benchmark'], 'MDP', 'MDP Cumulative Returns In-Sample')
SAA_IS_results = pd.concat([ERC_IS_results, SR_IS_results, MDP_IS_results], axis=1)
SAA_IS_results.to_latex('Output/SAA_IS_results.tex')

# =============================================================================
# 1.2
# =============================================================================

portfolio_OS_ERC = pd.DataFrame({'ERC': np.sum(np.multiply(returns_OS_price,np.transpose(res_ERC.x)), 1)})
portfolio_OS_SR = pd.DataFrame({'SR': np.sum(np.multiply(returns_OS_price,np.transpose(res_SR.x)), 1)})
portfolio_OS_MDP = pd.DataFrame({'MDP': np.sum(np.multiply(returns_OS_price,np.transpose(res_MDP.x)), 1)})

ERC_OS_results = perf(portfolio_OS_ERC['ERC'], benchmark_return_OS['OS Benchmark'], 'ERC', 'ERC Cumulative Returns Out-Sample')
SR_OS_results = perf(portfolio_OS_SR['SR'], benchmark_return_OS['OS Benchmark'], 'SR', 'SR Cumulative Returns Out-Sample')
MDP_OS_results = perf(portfolio_OS_MDP['MDP'], benchmark_return_OS['OS Benchmark'], 'MDP', 'MDP Cumulative Returns Out-Sample')
SAA_OS_results = pd.concat([ERC_OS_results, SR_OS_results, MDP_OS_results], axis=1)
SAA_OS_results.to_latex('Output/SAA_OS_results.tex')

# =============================================================================
# =============================================================================
# Part 2: TAA
# =============================================================================
# =============================================================================

# =============================================================================
# 2.1
# =============================================================================

df_carry = pd.read_excel("Data_QAM2.xlsx",sheet_name='Carry', engine='openpyxl')
df_carry.index = df_carry['Unnamed: 0']
df_carry.index.name = 'date'
del df_carry['Unnamed: 0']

###Value###

df_carry_insample = df_carry[df_carry.index <= pd.to_datetime('2010-12-31')]

df_carry_insample_Z = (df_carry_insample - np.mean(df_carry_insample))/np.std(df_carry_insample)
df_carry_insample_Z['median'] = df_carry_insample_Z.median(axis=1)
df_carry_insample_Z

df_carry_insample_Z_pos = df_carry_insample_Z.copy()
for value in df_carry_insample_Z.iloc[:,0:7].columns:
    list_value = []
    for i in range(len(df_carry_insample_Z)):
        if df_carry_insample_Z[value][i] > df_carry_insample_Z['median'][i]:
            list_value.append(1)
        else:
            list_value.append(-1)
    df_carry_insample_Z_pos[f'{value}'] = list_value
#df_carry_insample_Z_pos.to_csv('Carry_postion.csv')
df_carry_insample_Z_pos

position = df_carry_insample_Z_pos

position[position[position.columns] > 0]

del position['median']

weight_final = {}
for i in range(len(position)):
    num_pos = np.sum(np.array(position.values[i]) > 0)
    num_neg = np.sum(np.array(position.values[i]) < 0)
    array = []
    for j in range(len(position.values[i])):
        #print(position.values[i][j]/num_neg)
        if position.iloc[i,j] > 0:
           position.iloc[i,j] = position.iloc[i,j]/num_pos
        else:
            position.iloc[i,j] = position.iloc[i,j]/num_neg
    #print(array)
    #position.loc[i] = array
    #print(position.values[i])
position.to_csv('value_position.csv')

position_IS = position.copy()

portofolio_value = position.mul(returns_IS_price).sum(axis=1)

vol_budget = 0.02
vol_value_scaled = vol_budget/portofolio_value.std()

value_IS_scaled = pd.DataFrame({'Value': vol_value_scaled*portofolio_value})

###Momentum###

#Source: https://www.youtube.com/watch?v=dnrJ4zwCADM

#Calculate the returns over the past 11 months

returns_IS_past11 = (returns_IS_price+1).rolling(11).apply(np.prod) - 1
returns_IS_past11 = returns_IS_past11.dropna()

returns_IS_quantile = returns_IS_past11.T.apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"), axis=0).T

for i in returns_IS_quantile.columns:
    returns_IS_quantile.loc[returns_IS_quantile[i] == 4, i] = 0.5
    returns_IS_quantile.loc[returns_IS_quantile[i] == 0, i] = -0.5
    returns_IS_quantile.loc[returns_IS_quantile[i] == 1, i] = 0
    returns_IS_quantile.loc[returns_IS_quantile[i] == 2, i] = 0
    returns_IS_quantile.loc[returns_IS_quantile[i] == 3, i] = 0

np.sum(returns_IS_quantile, axis=1) == 0

weights_IS_mom = returns_IS_quantile.shift(1, axis = 0).dropna()

portofolio_mom = weights_IS_mom.multiply(returns_IS_price).sum(axis=1)

vol_budget = 0.02
vol_mom_scaled = vol_budget/portofolio_mom.std()

mom_IS_scaled = pd.DataFrame({'Momentum': vol_mom_scaled*portofolio_mom})

###Collect the Results###
value_IS_results = perf(value_IS_scaled['Value'], benchmark_return_IS['IS Benchmark'], 'Value', 'Value Cumulative Returns In-Sample')
mom_IS_results = perf(mom_IS_scaled['Momentum'], benchmark_return_IS['IS Benchmark'], 'Momentum', 'Momentum Cumulative Returns In-Sample')
ValMom_IS_results = pd.concat([value_IS_results, mom_IS_results], axis=1)
ValMom_IS_results.to_latex('Output/ValMom_IS_results.tex')


# =============================================================================
# 2.2
# =============================================================================

df_vix = pd.read_excel("Data_QAM2.xlsx",sheet_name='VIX', engine='openpyxl')
df_vix.set_index('Dates', inplace=True)

df_vix = (df_vix - df_vix.mean())/df_vix.std()
df_vix['Percentage Change'] = np.log(df_vix['VIX']/df_vix['VIX'].shift(1))
df_vix['Percentage Change'] = df_vix['Percentage Change'].replace(np.nan, 0)

df_vix_IS = df_vix.loc[(df_vix.index <= pd.to_datetime('2010-12-31'))]
df_vix_OS = df_vix.loc[(df_vix.index > pd.to_datetime('2010-12-31'))]

quantile = df_vix_IS.quantile(q=0.90)
df_vix_IS['Quantile'] = np.ones(df_vix_IS.shape[0])*quantile.VIX

plt.figure(figsize=(15,7))
plt.subplot(121)
plt.plot(df_vix_IS['Percentage Change'], 'r')
plt.title('Percentage Change of VIX In-Sample')
plt.subplot(122)
plt.plot(df_vix_IS['VIX'], label = 'Standardized VIX')
plt.plot(df_vix_IS['Quantile'], label = '90% Quantile')
plt.title('Standardized VIX In-Sample')
plt.legend(loc='upper left', frameon=True)
plt.savefig('Plot/VIX_IS.png')
plt.show()
plt.close()

plt.figure(figsize=(15,10))
plt.subplot(221)
corr_value_IS = sns.heatmap(pd.concat([df_vix_IS['VIX'], value_IS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve In-Sample')
plt.subplot(222)
corr_mom_IS = sns.heatmap(pd.concat([df_vix_IS['VIX'], mom_IS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve In-Sample')
plt.subplot(223)
plt.plot(df_vix_IS['VIX'].rolling(10).corr(value_IS_scaled))
plt.title('Rolling Covariance Value-VIX (10 Months) In-sample')
plt.subplot(224)
plt.plot(df_vix_IS['VIX'].rolling(10).corr(mom_IS_scaled))
plt.title('Rolling Covariance Momentum-VIX (10 Months) In-sample')
plt.savefig('Plot/VIX_analysis_IS.png')
plt.show()
plt.close()



"""
###Parametric 
df_vix_IS.loc[df_vix_IS['VIX'] <= quantile.VIX, 'Long/Short'] = 1 #Expansion
df_vix_IS.loc[df_vix_IS['VIX'] > quantile.VIX, 'Long/Short'] = -1 #Recession 

lambda_ra=3
denominator = np.cov(np.transpose(pd.concat([value_IS_scaled, mom_IS_scaled], axis=1).replace(np.nan, 0)))
z = repmat(df_vix_IS['Long/Short'], np.size(pd.concat([value_IS_scaled, mom_IS_scaled], axis=1).replace(np.nan, 0) ,1), 1)
z = np.transpose(z)
z = pd.DataFrame(z)
numerator = np.mean(np.multiply(z, pd.concat([value_IS_scaled, mom_IS_scaled], axis=1).replace(np.nan, 0)))

labels = ['Value', 'Momentum']

opt_weights=1/lambda_ra * np.matmul(np.linalg.inv(denominator), numerator)
weight_to_chart=opt_weights
plt.plot(labels, weight_to_chart,'ro', labels, weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation Alpha_t')
plt.ylabel('Theta')
plt.tight_layout()

z = repmat(df_vix_IS['Long/Short'], np.size(pd.concat([value_IS_scaled, mom_IS_scaled], axis=1).replace(np.nan, 0),1),1)
z = np.transpose(z)
z = pd.DataFrame(z)
prod = np.multiply(z,pd.concat([value_IS_scaled, mom_IS_scaled], axis=1).replace(np.nan, 0))
taa_IS = prod.dot(opt_weights)
taa_IS = taa_IS/(np.std(taa_IS)*np.power(12,0.5))*0.01

taa_IS = pd.DataFrame({'TAA_IS_VIX': taa_IS, 'Date': mom_IS_scaled.index})
taa_IS.set_index('Date', inplace=True)

perf(taa_IS, benchmark_return_IS['IS Benchmark'], 'Parametrics')
"""

###Simple Strategy###
"""
We need to long the momentum factor and the value factor when the VIX is low (at 90% quantile), and long the momentum factor and the value factor when the VIX is high (at 10% quantile).
"""
df_vix_IS.loc[df_vix_IS['VIX'] <= quantile.VIX, 'Value Position'] = 1 #Long Value
df_vix_IS.loc[df_vix_IS['VIX'] > quantile.VIX, 'Value Position'] = -1 #Short Value

df_vix_IS.loc[df_vix_IS['VIX'] <= quantile.VIX, 'Mom Position'] = 1 #Long Mom
df_vix_IS.loc[df_vix_IS['VIX'] > quantile.VIX, 'Mom Position'] = -1 #Short Mom

TAA_IS_VIX = pd.DataFrame({'Returns Strategy': value_IS_scaled['Value']*df_vix_IS['Value Position'] + mom_IS_scaled['Momentum']*df_vix_IS['Mom Position']}).replace(np.nan, 0)

TAA_IS_VIX_results = perf(TAA_IS_VIX['Returns Strategy'], benchmark_return_IS['IS Benchmark'], 'TAA_IS_VIX', 'TAA-VIX Cumulative Returns In-Sample (Simple Strategy)')


###Complex Strategy###
df_vix_IS.loc[(df_vix_IS['VIX'] <= quantile.VIX), 'Value Position'] = 0.5 #Long Value
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX)  & (df_vix_IS['Percentage Change'] >= 0), 'Value Position'] = 0 #Short Value
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX)  & (df_vix_IS['Percentage Change'] < 0), 'Value Position'] = 2 #Long Value

df_vix_IS.loc[df_vix_IS['VIX'] <= quantile.VIX, 'Mom Position'] = 0.5 #Long Mom
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX)  & (df_vix_IS['Percentage Change'] >= 0), 'Mom Position'] = 1 #Long Value
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX) & (df_vix_IS['Percentage Change'] < 0), 'Mom Position'] = -1 #Long Value

TAA_IS_VIX = pd.DataFrame({'Returns Strategy': value_IS_scaled['Value']*df_vix_IS['Value Position'] + mom_IS_scaled['Momentum']*df_vix_IS['Mom Position']}).replace(np.nan, 0)

TAA_IS_VIX_results = perf(TAA_IS_VIX['Returns Strategy'], benchmark_return_IS['IS Benchmark'], 'TAA_IS_VIX', 'TAA-VIX Cumulative Returns In-Sample (Complex Strategy)')

TAA_IS = pd.concat([value_IS_results, mom_IS_results, TAA_IS_VIX_results], axis=1)
TAA_IS.to_latex('Output/TAA_IS.tex')

###Information Ratio###
""" Check the end of code. 
#https://www.jstor.org/stable/4480091?seq=1#metadata_info_tab_contents
#https://en.wikipedia.org/wiki/Information_ratio

ER_IS = TAA_IS_VIX['Returns Strategy'] - benchmark_return_IS['IS Benchmark']

IR_IS = (np.mean(ER_IS, 0)*12)/np.std(ER_IS, 0)*np.power(12,0.5)
"""
# =============================================================================
# 2.3
# =============================================================================

###Value###
df_carry_outsample = df_carry[df_carry.index > pd.to_datetime('2010-12-31')]

df_carry_outsample_Z = (df_carry_outsample - np.mean(df_carry_outsample))/np.std(df_carry_outsample)
df_carry_outsample_Z['median'] = df_carry_outsample_Z.median(axis=1)
df_carry_outsample_Z

df_carry_outsample_Z_pos = df_carry_outsample_Z.copy()
for value in df_carry_outsample_Z.iloc[:,0:7].columns:
    list_value = []
    for i in range(len(df_carry_outsample_Z)):
        if df_carry_outsample_Z[value][i] > df_carry_outsample_Z['median'][i]:
            list_value.append(1)
        else:
            list_value.append(-1)
    df_carry_outsample_Z_pos[f'{value}'] = list_value
#df_carry_outsample_Z_pos.to_csv('Carry_postion.csv')
df_carry_outsample_Z_pos

position = df_carry_outsample_Z_pos
position[position[position.columns] > 0]

del position['median']

weight_final = {}
for i in range(len(position)):
    num_pos = np.sum(np.array(position.values[i]) > 0)
    num_neg = np.sum(np.array(position.values[i]) < 0)
    array = []
    for j in range(len(position.values[i])):
        #print(position.values[i][j]/num_neg)
        if position.iloc[i,j] > 0:
           position.iloc[i,j] = position.iloc[i,j]/num_pos
        else:
            position.iloc[i,j] = position.iloc[i,j]/num_neg
    #print(array)
    #position.loc[i] = array
    #print(position.values[i])
position.to_csv('value_position.csv')

position_OS = position.copy()

portofolio_value = position.mul(returns_OS_price).sum(axis=1)

vol_budget = 0.02
vol_value_scaled = vol_budget/portofolio_value.std()

value_OS_scaled = pd.DataFrame({'Value': vol_value_scaled*portofolio_value})

###Momentum###

#Source: https://www.youtube.com/watch?v=dnrJ4zwCADM

#Calculate the returns over the past 11 months
returns_past11 = (returns_price+1).rolling(11).apply(np.prod) - 1
returns_past11 = returns_past11.dropna()

returns_quantile = returns_past11.T.apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"), axis=0).T

for i in returns_quantile.columns:
    returns_quantile.loc[returns_quantile[i] == 4, i] = 0.5
    returns_quantile.loc[returns_quantile[i] == 0, i] = -0.5
    returns_quantile.loc[returns_quantile[i] == 1, i] = 0
    returns_quantile.loc[returns_quantile[i] == 2, i] = 0
    returns_quantile.loc[returns_quantile[i] == 3, i] = 0

np.sum(returns_quantile, axis=1) == 0

weights_mom = returns_quantile.shift(1, axis = 0).dropna()

portofolio_mom_OS = weights_mom.multiply(returns_OS_price).sum(axis=1)
portofolio_mom_OS = pd.DataFrame({'Momentum': portofolio_mom_OS})
portofolio_mom_OS =  portofolio_mom_OS.loc[(portofolio_mom_OS != 0).any(1)]

vol_budget = 0.02
vol_mom_scaled = vol_budget/portofolio_mom_OS.std()

mom_OS_scaled = vol_mom_scaled*portofolio_mom_OS

###Collect the Results###
value_OS_results = perf(value_OS_scaled['Value'], benchmark_return_OS['OS Benchmark'], 'Value', 'Value Cumulative Returns Out-Sample')
mom_OS_results = perf(mom_OS_scaled['Momentum'], benchmark_return_OS['OS Benchmark'], 'Momentum', 'Momentum Cumulative Returns Out-Sample')
ValMom_OS_results = pd.concat([value_OS_results, mom_OS_results], axis=1)
ValMom_OS_results.to_latex('Output/ValMom_OS_results.tex')

###Out-Sample VIX###

quantile = df_vix_IS.quantile(q=0.90)
df_vix_OS['Quantile'] = np.ones(df_vix_OS.shape[0])*quantile.VIX

plt.figure(figsize=(15,7))
plt.subplot(121)
plt.plot(df_vix_OS['Percentage Change'], 'r')
plt.title('Percentage Change of VIX Out-Sample')
plt.subplot(122)
plt.plot(df_vix_OS['VIX'], 'b', label = 'Standardized VIX')
#plt.plot(df_vix_IS['Quantile'], label = '90% Quantile')
plt.title('Standardized VIX In-Sample')
#plt.legend(loc='upper left', frameon=True)
plt.savefig('Plot/VIX_OS.png')
plt.show()
plt.close()

plt.figure(figsize=(15,10))
plt.subplot(221)
corr_value_OS = sns.heatmap(pd.concat([df_vix_OS['VIX'], value_OS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve Out-Sample')
plt.subplot(222)
corr_mom_OS = sns.heatmap(pd.concat([df_vix_OS['VIX'], mom_OS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve Out-Sample')
plt.subplot(223)
plt.plot(df_vix_OS['VIX'].rolling(10).corr(value_OS_scaled))
plt.title('Rolling Covariance Value-VIX (10 Months) Out-sample')
plt.subplot(224)
plt.plot(df_vix_OS['VIX'].rolling(10).corr(mom_OS_scaled))
plt.title('Rolling Covariance Momentum-VIX (10 Months) Out-sample')
plt.savefig('Plot/VIX_analysis_OS.png')
plt.show()
plt.close()

"""
###Parametric 
df_vix_OS.loc[df_vix_OS['VIX'] <= quantile.VIX, 'Long/Short'] = 1 #Expansion
df_vix_OS.loc[df_vix_OS['VIX'] > quantile.VIX, 'Long/Short'] = -1 #Recession 

lambda_ra=3
denominator = np.cov(np.transpose(pd.concat([value_OS_scaled, mom_OS_scaled], axis=1).replace(np.nan, 0)))
z = repmat(df_vix_OS['Long/Short'], np.size(pd.concat([value_OS_scaled, mom_OS_scaled], axis=1).replace(np.nan, 0) ,1), 1)
z = np.transpose(z)
z = pd.DataFrame(z)
numerator = np.mean(np.multiply(z, pd.concat([value_OS_scaled, mom_OS_scaled], axis=1).replace(np.nan, 0)))

labels = ['Value', 'Momentum']

opt_weights=1/lambda_ra * np.matmul(np.linalg.inv(denominator), numerator)
weight_to_chart=opt_weights
plt.plot(labels, weight_to_chart,'ro', labels, weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation Alpha_t')
plt.ylabel('Theta')
plt.tight_layout()

z = repmat(df_vix_OS['Long/Short'], np.size(pd.concat([value_OS_scaled, mom_OS_scaled], axis=1).replace(np.nan, 0) ,1), 1)
z = np.transpose(z)
z = pd.DataFrame(z)
prod = np.multiply(z,pd.concat([value_OS_scaled, mom_OS_scaled], axis=1).replace(np.nan, 0))
taa_OS = prod.dot(opt_weights)
taa_OS = taa_OS/(np.std(taa_OS)*np.power(12,0.5))*0.01

taa_OS = pd.DataFrame({'TAA_IS_VIX': taa_OS, 'Date': mom_OS_scaled.index})
taa_OS.set_index('Date', inplace=True)

perf(taa_OS, benchmark_return_OS['OS Benchmark'], 'Parametrics')
"""

###Simple Strategy###
"""
We need to long the momentum factor and the value factor when the VIX is low (at 90% quantile), and long the momentum factor and the value factor when the VIX is high (at 10% quantile).
"""
df_vix_OS.loc[df_vix_OS['VIX'] <= quantile.VIX, 'Value Position'] = 1 #Long Value
df_vix_OS.loc[df_vix_OS['VIX'] > quantile.VIX, 'Value Position'] = -1 #Short Value

df_vix_OS.loc[df_vix_OS['VIX'] <= quantile.VIX, 'Mom Position'] = 1 #Long Mom
df_vix_OS.loc[df_vix_OS['VIX'] > quantile.VIX, 'Mom Position'] = 1 #Short Value

TAA_OS_VIX = pd.DataFrame({'Returns Strategy': value_OS_scaled['Value']*df_vix_OS['Value Position'] + mom_OS_scaled['Momentum']*df_vix_OS['Mom Position']}).replace(np.nan, 0)
TAA_OS_VIX_results = perf(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'], 'TAA_OS_VIX','TAA Cumulative Returns Out-Sample (Simple Strategy)')

###Complex Strategy###
"""
We need to short the momentum factor/long the value factor when the VIX is low (at 90% quantile), and long the momentum factor/short the value factor when the VIX is high (at 10% quantile).
"""

df_vix_OS.loc[(df_vix_OS['VIX'] <= quantile.VIX), 'Value Position'] = 0.5 #Long Value
df_vix_OS.loc[(df_vix_OS['VIX'] > quantile.VIX)  & (df_vix_OS['Percentage Change'] >= 0), 'Value Position'] = 0 #Short Value
df_vix_OS.loc[(df_vix_OS['VIX'] > quantile.VIX)  & (df_vix_OS['Percentage Change'] < 0), 'Value Position'] = 2 #Long Value

df_vix_OS.loc[df_vix_OS['VIX'] <= quantile.VIX, 'Mom Position'] = 0.5 #Long Mom
df_vix_OS.loc[(df_vix_OS['VIX'] > quantile.VIX)  & (df_vix_OS['Percentage Change'] >= 0), 'Mom Position'] = 1 #Long Value
df_vix_OS.loc[(df_vix_OS['VIX'] > quantile.VIX) & (df_vix_OS['Percentage Change'] < 0), 'Mom Position'] = -1 #Long Value

TAA_OS_VIX = pd.DataFrame({'Returns Strategy': value_OS_scaled['Value']*df_vix_OS['Value Position'] + mom_OS_scaled['Momentum']*df_vix_OS['Mom Position']}).replace(np.nan, 0)

TAA_OS_VIX_results = perf(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'], 'TAA_OS_VIX','TAA Cumulative Returns Out-Sample (Complex Strategy)')

###Complex Strategy V2 (Adaptive Quantile)###

df_vix_OS['Value Position'] = 0
df_vix_OS['Mom Position'] = 0
for i in range(0, df_vix_OS.shape[0]):
    quantile = df_vix.iloc[:df_vix_IS.shape[0]+1+i, 0].quantile(q=0.90)
    print(quantile)
    if (df_vix_OS.iloc[i, 0] <= quantile):
        df_vix_OS.iloc[i, 3] = 0.5
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] >= 0):
        df_vix_OS.iloc[i, 3] = 0
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] < 0):
        df_vix_OS.iloc[i, 3] = 2
    else:
        pass
    
    if (df_vix_OS.iloc[i, 0] <= quantile):
        df_vix_OS.iloc[i, 4] = 0.5
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] >= 0):
        df_vix_OS.iloc[i, 4] = 1
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] < 0):
        df_vix_OS.iloc[i, 4] = -1 
    else:
        pass
        
TAA_OS_VIX = pd.DataFrame({'Returns Strategy': value_OS_scaled['Value']*df_vix_OS['Value Position'] + mom_OS_scaled['Momentum']*df_vix_OS['Mom Position']}).replace(np.nan, 0)

TAA_OS_VIX_results = perf(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'], 'TAA_OS_VIX', 'TAA Cumulative Returns Out-Sample')

TAA_OS = pd.concat([value_OS_results, mom_OS_results, TAA_OS_VIX_results], axis=1)
TAA_OS.to_latex('Output/TAA_OS.tex')

# =============================================================================
# =============================================================================
# Part 3
# =============================================================================
# =============================================================================

"""
###ERC Allocation###

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

cons = ({'type':'eq', 'fun': lambda x:sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]-0.01},
      {'type':'eq', 'fun': lambda x: x[2]},
      {'type':'eq', 'fun': lambda x: np.mean(np.sum(np.multiply(returns_IS_price,np.transpose(x)), 1),0)*12 - SAA_IS_results['ERC'].Mean},
      {'type':'eq', 'fun': lambda x: np.std(np.sum(np.multiply(returns_IS_price,np.transpose(x)), 1),0)*np.power(12,0.5) - SAA_IS_results['ERC'].Volatility})

Bounds = [(0 , 1) for i in range(0,7)]

res_ERC_cons = minimize(ERC, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

#Plot the optimal weights using ERC
weight_to_chart=np.array(res_ERC_cons.x)
plt.plot(labels, weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation ERC')
plt.ylabel('Ptf Weight')
plt.tight_layout()

#MCR for Max SR
MCR_chart=MCR_calc(res_ERC_cons.x, returns_IS_price);
MCR_chart=np.array(MCR_chart);
plt.plot(labels, MCR_chart, 'ro', labels, MCR_chart*0, 'b-')
plt.xticks(rotation=90)
plt.title('MCR ERC')
plt.ylabel('MCR')
plt.tight_layout()

#Test for the MCR/Risk ratio
ratio = res_ERC_cons.x*MCR_chart
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('alpha x MCR')
plt.ylabel('Ratio')
plt.tight_layout()

###Sharp Ratio Allocation###

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

cons=({'type':'eq', 'fun': lambda x:sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]-0.01},
      {'type':'eq', 'fun': lambda x: x[2]},
      {'type':'eq', 'fun': lambda x: np.mean(np.sum(np.multiply(returns_IS_price,np.transpose(x)), 1),0)*12 - SAA_IS_results['SR'].Mean},
      {'type':'eq', 'fun': lambda x: np.std(np.sum(np.multiply(returns_IS_price,np.transpose(x)), 1),0)*np.power(12,0.5) - SAA_IS_results['SR'].Volatility})

Bounds= [(0 , 1) for i in range(0,7)]
#Bounds = [(0 , 1), (0.01 , 1), (0.01 , 1), (0 , 1), (0 , 1), (0 , 1), (0 , 1)]

res_SR_cons = minimize(Max_SR, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

#Plot the optimal weights using Sharpe Ratio
weight_to_chart=np.array(res_SR_cons.x)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation SR')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_SR_cons.x, returns_IS_price);
MCR_chart=np.array(MCR_chart);
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR SR')
plt.ylabel('MCR')
plt.tight_layout()

# Test for the Risk/Reward ratio
reward=np.mean(returns_IS_price, 0)
ratio=MCR_chart/reward
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('Excess Return to MCR Ratio')
plt.ylabel('Ratio')
plt.tight_layout()

###Most-Diversified Portfolio###

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

cons=({'type':'eq', 'fun': lambda x:sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]-0.01},
      {'type':'eq', 'fun': lambda x: x[2]},
      {'type':'eq', 'fun': lambda x: np.mean(np.sum(np.multiply(returns_IS_price,np.transpose(x)), 1),0)*12 - SAA_IS_results['MDP'].Mean},
      {'type':'eq', 'fun': lambda x: np.std(np.sum(np.multiply(returns_IS_price,np.transpose(x)), 1),0)*np.power(12,0.5) - SAA_IS_results['MDP'].Volatility})

Bounds= [(0 , 1) for i in range(0,7)]
#Bounds = [(0 , 1), (0.01 , 1), (0.01 , 1), (0 , 1), (0 , 1), (0 , 1), (0 , 1)]

res_MDP_cons = minimize(Max_DP, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

weight_to_chart=np.array(res_MDP_cons.x)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation Max Div')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_MDP_cons.x, returns_IS_price);
MCR_chart=np.array(MCR_chart);
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR MaxDiv')
plt.ylabel('MCR')
plt.tight_layout()

#Test for MCR/Risk ratio
MCR_risk = MCR_chart/np.std(returns_IS_price)
plt.plot(labels, MCR_risk, 'ro', labels, MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR/Risk Ratio')
plt.ylabel('Ratio')
plt.tight_layout()

###Determine the performances of the IS  optimal portfolio allocatio###
portfolio_IS_ERC_cons = pd.DataFrame({'ERC': np.sum(np.multiply(returns_IS_price,np.transpose(res_ERC_cons.x)), 1)})
portfolio_IS_SR_cons = pd.DataFrame({'SR': np.sum(np.multiply(returns_IS_price,np.transpose(res_SR_cons.x)), 1)})
portfolio_IS_MDP_cons = pd.DataFrame({'MDP': np.sum(np.multiply(returns_IS_price,np.transpose(res_MDP_cons.x)), 1)})

ERC_IS_results_cons = perf(portfolio_IS_ERC_cons['ERC'], benchmark_return_IS['IS Benchmark'], 'ERC')
SR_IS_results_cons = perf(portfolio_IS_SR_cons['SR'], benchmark_return_IS['IS Benchmark'], 'SR')
MDP_IS_results_cons = perf(portfolio_IS_MDP_cons['MDP'], benchmark_return_IS['IS Benchmark'], 'MDP')
SAA_IS_results_cons = pd.concat([ERC_IS_results_cons, SR_IS_results_cons, MDP_IS_results_cons], axis=1)

###Determine the performances of the IS  optimal portfolio allocatio###
portfolio_OS_ERC_cons = pd.DataFrame({'ERC': np.sum(np.multiply(returns_OS_price,np.transpose(res_ERC_cons.x)), 1)})
portfolio_OS_SR_cons = pd.DataFrame({'SR': np.sum(np.multiply(returns_OS_price,np.transpose(res_SR_cons.x)), 1)})
portfolio_OS_MDP_cons = pd.DataFrame({'MDP': np.sum(np.multiply(returns_OS_price,np.transpose(res_MDP_cons.x)), 1)})

ERC_OS_results_cons = perf(portfolio_OS_ERC_cons['ERC'], benchmark_return_OS['OS Benchmark'], 'ERC')
SR_OS_results_cons = perf(portfolio_OS_SR_cons['SR'], benchmark_return_OS['OS Benchmark'], 'SR')
MDP_OS_results_cons = perf(portfolio_OS_MDP_cons['MDP'], benchmark_return_OS['OS Benchmark'], 'MDP')
SAA_OS_results_cons = pd.concat([ERC_OS_results_cons, SR_OS_results_cons, MDP_OS_results_cons], axis=1)
"""

###Collecting All Weights###
SAA_weights_IS = pd.DataFrame({'World Equities': np.ones(returns_IS_price.shape[0])*res_ERC.x[0],
                                   'World Bonds': np.ones(returns_IS_price.shape[0])*res_ERC.x[1],
                                   'US Investment Grade': np.ones(returns_IS_price.shape[0])*res_ERC.x[2],
                                   'US High Yield': np.ones(returns_IS_price.shape[0])*res_ERC.x[3],
                                   'Gold': np.ones(returns_IS_price.shape[0])*res_ERC.x[4],
                                   'Energy': np.ones(returns_IS_price.shape[0])*res_ERC.x[5],
                                   'Copper': np.ones(returns_IS_price.shape[0])*res_ERC.x[6]
                                   }, index = returns_IS_price.index)

SAA_weights_OS = pd.DataFrame({'World Equities': np.ones(returns_OS_price.shape[0])*res_ERC.x[0],
                                   'World Bonds': np.ones(returns_OS_price.shape[0])*res_ERC.x[1],
                                   'US Investment Grade': np.ones(returns_OS_price.shape[0])*res_ERC.x[2],
                                   'US High Yield': np.ones(returns_OS_price.shape[0])*res_ERC.x[3],
                                   'Gold': np.ones(returns_OS_price.shape[0])*res_ERC.x[4],
                                   'Energy': np.ones(returns_OS_price.shape[0])*res_ERC.x[5],
                                   'Copper': np.ones(returns_OS_price.shape[0])*res_ERC.x[6]
                                   }, index = returns_OS_price.index)

value_weights_IS = position_IS

value_weights_OS = position_OS

mom_weights_IS = weights_IS_mom

mom_weights_OS = weights_mom.iloc[(weights_mom.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

VIX_weights_IS = pd.DataFrame({'Value Position': df_vix_IS['Value Position'], 
                               'Mom Position': df_vix_IS['Mom Position']})

VIX_weights_OS = pd.DataFrame({'Value Position': df_vix_OS['Value Position'], 
                               'Mom Position': df_vix_OS['Mom Position']})

TAA_weights_IS = value_weights_IS.multiply(VIX_weights_IS['Value Position'], axis='index') +  mom_weights_IS.multiply(VIX_weights_IS['Mom Position'], axis='index').replace(np.nan, 0)

TAA_weights_OS = (value_weights_OS.multiply(VIX_weights_OS['Value Position'], axis='index') +  mom_weights_OS.multiply(VIX_weights_OS['Mom Position'], axis='index').replace(np.nan, 0))

weight_target_IS = SAA_weights_IS + TAA_weights_IS

weight_target_OS = SAA_weights_OS + TAA_weights_OS

###Tracking Error Formula###
def TE(x):
    global weight_target
    global sigma
    weight_ptf = x
    diff_alloc = weight_ptf - weight_target
    temp =  np.matmul(diff_alloc.T, sigma)
    var_month = np.matmul(temp, diff_alloc)
    vol_month = np.power(var_month, 0.5)
    return vol_month*np.power(12, 0.5)

sigma_IS = returns_IS_price.cov().values
sigma_OS = returns_OS_price.cov().values

###Ex Ante Tracking Error & Replication Portfolion In-Sample###
cons = ({'type':'eq', 'fun': lambda weight_ptf: sum(weight_ptf)-1})

Bounds = [(-10 , 10), (-10 , 10), (0 , 0), (-10 , 10), (-10 , 10), (-10 , 10), (-10 , 10)]

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

weight_opt_IS = []
TE_ReplicationvsTarget_IS = []
for i in range(0, weight_target_IS.shape[0]):
    x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001
    weight_target = weight_target_IS.iloc[i, :].values
    sigma = sigma_IS
    opt_TE_IS = minimize(TE, x0, method='trust-constr', bounds=Bounds, constraints=cons, options={'disp': True}) #Powell/trust-constr works
    weight_opt_IS.append(opt_TE_IS.x)
    TE_ReplicationvsTarget_IS.append(opt_TE_IS.fun)
 
TE_ReplicationvsTarget_IS = pd.DataFrame({'TE_IS': TE_ReplicationvsTarget_IS}, index = weight_target_IS.index)
weight_rep_IS = np.array(weight_opt_IS)
weight_rep_IS = pd.DataFrame({'World Equities': weight_rep_IS[:, 0],
                              'World Bonds': weight_rep_IS[:, 1],
                              'US Investment Grade': weight_rep_IS[:, 2],
                              'US High Yield': weight_rep_IS[:, 3],
                              'Gold': weight_rep_IS[:, 4],
                              'Energy': weight_rep_IS[:, 5],
                              'Copper': weight_rep_IS[:, 6]}, index = weight_target_IS.index)

return_target_IS  = np.sum(np.multiply(returns_IS_price, weight_target_IS), axis=1)
return_rep_IS  = np.sum(np.multiply(returns_IS_price, weight_rep_IS), axis=1)

perf_target_IS = perf(return_target_IS, benchmark_return_IS['IS Benchmark'], 'Model', 'Target Portfolio In-Sample')
perf_rep_IS = perf(return_rep_IS, benchmark_return_IS['IS Benchmark'], 'Replication', 'Replication Portfolio In-Sample')

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot((return_target_IS+1).cumprod()*100, 'b', label='Model Portfolio')
plt.plot((return_rep_IS+1).cumprod()*100, 'r', label='Replication Portfolio')
plt.title('In-Sample Cumulative Returns')
plt.legend(loc='upper left', frameon=True)
plt.subplot(122)
plt.plot(TE_ReplicationvsTarget_IS)
plt.title('Ex-Ante Tracking Error between Real Ptf vs. SAA+TAA')
plt.savefig('Plot/target_replication_is.png')
plt.show()
plt.close()

###Ex Post Tracking Error & Replication Portfolio Out-Sample###
cons = ({'type':'eq', 'fun': lambda x: sum(x)-1})

Bounds = [(-10 , 10), (-10 , 10), (0 , 0), (-10 , 10), (-10 , 10), (-10 , 10), (-10 , 10)]

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

weight_opt_OS = []
TE_ReplicationvsTarget_OS = []
for i in range(0, weight_target_OS.shape[0]):
    x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001
    weight_target = weight_target_OS.iloc[i, :].values
    sigma = sigma_IS
    opt_TE_OS = minimize(TE, x0, method='trust-constr', bounds=Bounds, constraints=cons, options={'disp': True}) #Powell works
    weight_opt_OS.append(opt_TE_OS.x)
    TE_ReplicationvsTarget_OS.append(opt_TE_OS.fun)
 
TE_ReplicationvsTarget_OS = pd.DataFrame({'TE_OS': TE_ReplicationvsTarget_OS}, index = weight_target_OS.index)
weight_rep_OS = np.array(weight_opt_OS)
weight_rep_OS = pd.DataFrame({'World Equities': weight_rep_OS[:, 0],
                              'World Bonds': weight_rep_OS[:, 1],
                              'US Investment Grade': weight_rep_OS[:, 2],
                              'US High Yield': weight_rep_OS[:, 3],
                              'Gold': weight_rep_OS[:, 4],
                              'Energy': weight_rep_OS[:, 5],
                              'Copper': weight_rep_OS[:, 6]}, index = weight_target_OS.index)

return_target_OS  = np.sum(np.multiply(returns_OS_price, weight_target_OS), axis=1)
return_rep_OS  = np.sum(np.multiply(returns_OS_price, weight_rep_OS), axis=1)

perf_target_OS = perf(return_target_OS, benchmark_return_OS['OS Benchmark'], 'Model', 'Target Portfolio Out-Sample')
perf_rep_OS = perf(return_rep_OS, benchmark_return_OS['OS Benchmark'], 'Replication', 'Replication Portfolio Out-Sample')
perf_target_rep_OS = pd.concat([perf_target_OS, perf_rep_OS], axis=1)
perf_target_rep_OS.to_latex('Output/perf_target_rep_OS.tex')

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot((return_target_OS+1).cumprod()*100, 'b', label='Model Portfolio')
plt.plot((return_rep_OS+1).cumprod()*100, 'r', label='Replication Portfolio')
plt.title('Out-of-Sample Cumulative Returns')
plt.legend(loc='upper left', frameon=True)
plt.subplot(122)
plt.plot(TE_ReplicationvsTarget_OS)
plt.title('Ex-Post Tracking Error between Real Ptf vs. SAA+TAA')
plt.savefig('Plot/target_replication_os.png')
plt.show()
plt.close()

###Tracking Error###
"""Between SAA and Benchmark"""
Tot_TE_SAAvsBenchmark_IS = []
for i in range(0, SAA_weights_IS.shape[0]):
    weight_target = benchmark_return_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsBenchmark_IS = TE(SAA_weights_IS.iloc[i, :].values)
    Tot_TE_SAAvsBenchmark_IS.append(TE_SAAvsBenchmark_IS)
    
Tot_TE_SAAvsBenchmark_OS = []
for i in range(0, SAA_weights_OS.shape[0]):
    weight_target = benchmark_return_OS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsBenchmark_OS = TE(SAA_weights_OS.iloc[i, :].values)
    Tot_TE_SAAvsBenchmark_OS.append(TE_SAAvsBenchmark_OS)
 
TE_SAAvsBenchmark_IS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsBenchmark_IS}, index = benchmark_return_IS.index)
TE_SAAvsBenchmark_OS = pd.DataFrame({'TE_OS': Tot_TE_SAAvsBenchmark_OS}, index = benchmark_return_OS.index)

plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_SAAvsBenchmark_IS, 'b')
plt.title('Ex-Ante Tracking Error between SAA vs. SAA+TAA')
plt.subplot(122)
plt.plot(TE_SAAvsBenchmark_OS, 'r')
plt.title('Ex-Post Tracking Error between SAA vs. SAA+TAA')
plt.savefig('Plot/TE_SAAvsBenchmark.png')
plt.show() 
plt.close()

"""Between SAA+TAA and SAA"""
Tot_TE_SAAvsTarget_IS = []
for i in range(0, weight_target_IS.shape[0]):
    weight_target = SAA_weights_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTarget_IS = TE(weight_target_IS.iloc[i, :].values)
    Tot_TE_SAAvsTarget_IS.append(TE_SAAvsTarget_IS)
 
Tot_TE_SAAvsTarget_OS = []
for i in range(0, weight_target_OS.shape[0]):
    weight_target = SAA_weights_OS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTarget_OS = TE(weight_target_OS.iloc[i, :].values)
    Tot_TE_SAAvsTarget_OS.append(TE_SAAvsTarget_OS)
 
TE_SAAvsTarget_IS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsTarget_IS}, index = weight_target_IS.index)
TE_SAAvsTarget_OS = pd.DataFrame({'TE_OS': Tot_TE_SAAvsTarget_OS}, index = weight_target_OS.index)

plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_SAAvsTarget_IS, 'b')
plt.title('Ex-Ante Tracking Error between SAA vs. SAA+TAA')
plt.subplot(122)
plt.plot(TE_SAAvsTarget_OS, 'r')
plt.title('Ex-Post Tracking Error between SAA vs. SAA+TAA')
plt.savefig('Plot/TE_SAAvsTarget.png')
plt.show() 
plt.close()

"""Collect All Output"""
TE_annualized_output = pd.DataFrame({'In-Sample': [TE_SAAvsBenchmark_IS['TE_IS'].std(axis=0)*(12**0.5), TE_SAAvsTarget_IS['TE_IS'].std(axis=0)*(12**0.5), TE_ReplicationvsTarget_IS['TE_IS'].std(axis=0)*(12**0.5)],
                          'Out-Sample': [TE_SAAvsBenchmark_OS['TE_OS'].std(axis=0)*(12**0.5), TE_SAAvsTarget_OS['TE_OS'].std(axis=0)*(12**0.5), TE_ReplicationvsTarget_OS['TE_OS'].std(axis=0)*(12**0.5)]},
                            index=['SAA vs. Benchmark', 'TAA & SAA vs. SAA', 'Replication vs. TAA & SAA '])
TE_annualized_output.to_latex('Output/TE_annualized_output.tex')

###Information Ratio###
def info_ratio(return_p, return_b):
    excess = return_p - return_b
    return (excess.mean(axis=0)*12)/(excess.std(axis=0)*(12**0.5))

"""Between Benchmark and SAA"""
IR_SAAvsBenchmark_IS = info_ratio(portfolio_IS_ERC['ERC'], benchmark_return_IS['IS Benchmark'])
IR_SAAvsBenchmark_OS = info_ratio(portfolio_IS_ERC['ERC'], benchmark_return_OS['OS Benchmark'])

"""Between SAA+TAA and SAA"""
IR_TargetvsSAA_IS = info_ratio(return_target_IS, portfolio_IS_ERC['ERC'])
IR_TargetvsSAA_OS = info_ratio(return_target_OS, portfolio_OS_ERC['ERC'])

"""Between Replication Portfolion and SAA+TAA (i.e. Target Portfolio)"""
IR_ReplicationvsTarget_IS = info_ratio(return_rep_IS, return_target_IS)
IR_ReplicationvsTarget_OS = info_ratio(return_rep_OS, return_target_OS)

"""Collect All Output"""
IR_output = pd.DataFrame({'In-Sample': [IR_SAAvsBenchmark_IS, IR_TargetvsSAA_IS, IR_ReplicationvsTarget_IS],
                          'Out-Sample': [IR_SAAvsBenchmark_OS, IR_TargetvsSAA_OS, IR_ReplicationvsTarget_OS]},
                         index=['SAA vs. Benchmark', 'TAA & SAA vs. SAA', 'Replication vs. TAA & SAA '])
IR_output.to_latex('Output/IR_output.tex')


###Allocation/Selection Performance Attribution In-Sample###

#weight for each type of asset (target)
weights_bonds_target_IS = weight_target_IS['World Bonds'] + weight_target_IS['US Investment Grade'] + weight_target_IS['US High Yield']
weights_equities_target_IS = weight_target_IS['World Equities']
weights_commodities_target_IS = weight_target_IS['Gold'] +  weight_target_IS['Energy'] + weight_target_IS['Copper']

#Weight for each type of asset (replicatio/portfolio)
weights_bonds_rep_IS = weight_rep_IS['World Bonds'] + weight_rep_IS['US Investment Grade'] + weight_rep_IS['US High Yield']
weights_equities_rep_IS = weight_rep_IS['World Equities']
weights_commodities_rep_IS = weight_rep_IS['Gold'] +  weight_rep_IS['Energy'] + weight_rep_IS['Copper']

#Performance for each type of asset (target)
perf_bonds_target_IS = (np.multiply(returns_IS_price['World Bonds'], weight_target_IS['World Bonds']) + np.multiply(returns_IS_price['US Investment Grade'], weight_target_IS['US Investment Grade']) + np.multiply(returns_IS_price['US High Yield'], weight_target_IS['US High Yield']))/weights_bonds_target_IS
perf_equities_target_IS = (np.multiply(returns_IS_price['World Equities'], weight_target_IS['World Equities']))/weights_equities_target_IS
perf_commodities_target_IS = (np.multiply(returns_IS_price['Gold'], weight_target_IS['Gold']) + np.multiply(returns_IS_price['Energy'], weight_target_IS['Energy']) + np.multiply(returns_IS_price['Copper'], weight_target_IS['Copper']))/weights_commodities_target_IS

#Performance for each type of asset (replicatio/portfolio)
perf_bonds_rep_IS = (np.multiply(returns_IS_price['World Bonds'], weight_rep_IS['World Bonds']) + np.multiply(returns_IS_price['US Investment Grade'], weight_rep_IS['US Investment Grade']) + np.multiply(returns_IS_price['US High Yield'], weight_rep_IS['US High Yield']))/weights_bonds_rep_IS
perf_equities_rep_IS = (np.multiply(returns_IS_price['World Equities'], weight_rep_IS['World Equities']))/weights_equities_rep_IS
perf_commodities_rep_IS = (np.multiply(returns_IS_price['Gold'], weight_rep_IS['Gold']) + np.multiply(returns_IS_price['Energy'], weight_rep_IS['Energy']) + np.multiply(returns_IS_price['Copper'], weight_rep_IS['Copper']))/weights_commodities_rep_IS

R_IS = weights_bonds_rep_IS*perf_bonds_rep_IS +  weights_equities_rep_IS*perf_equities_rep_IS + weights_commodities_rep_IS*perf_commodities_rep_IS

B_IS = weights_bonds_target_IS*perf_bonds_target_IS + weights_equities_target_IS*perf_equities_target_IS + weights_commodities_target_IS*perf_commodities_target_IS

R_S_IS = weights_bonds_target_IS*perf_bonds_rep_IS + weights_equities_target_IS*perf_equities_rep_IS + weights_commodities_target_IS*perf_commodities_rep_IS

B_S_IS = weights_bonds_rep_IS*perf_bonds_target_IS + weights_equities_rep_IS*perf_equities_target_IS + weights_commodities_rep_IS*perf_commodities_target_IS

plt.figure(figsize=(16,7))
plt.subplot(131)
#interaction_IS = R_IS - R_S_IS - B_S_IS + B_S_IS
interaction_IS = R_IS - R_S_IS - B_S_IS + B_IS
plt.plot(interaction_IS, 'r')
plt.title('Monthly Interaction Effect In-Sample')

plt.subplot(132)
#allocation_IS = B_IS - R_S_IS
allocation_IS = B_S_IS - B_IS 
plt.plot(allocation_IS, 'b')
plt.title('Monthly Allocation Effect In-Sample')

plt.subplot(133)
#selection_IS = B_IS - B_S_IS
selection_IS = R_S_IS - B_IS 
plt.plot(selection_IS, 'g')
plt.title('Monthly Selection Effect In-Sample')
plt.show()
plt.close()

###Allocation/Selection Performance Attribution Out-Of-Sample###

#weight for each type of asset (target)
weights_bonds_target_OS = weight_target_OS['World Bonds'] + weight_target_OS['US Investment Grade'] + weight_target_OS['US High Yield']
weights_equities_target_OS = weight_target_OS['World Equities']
weights_commodities_target_OS = weight_target_OS['Gold'] +  weight_target_OS['Energy'] + weight_target_OS['Copper']

#Weight for each type of asset (replicatio/portfolio)
weights_bonds_rep_OS = weight_rep_OS['World Bonds'] + weight_rep_OS['US Investment Grade'] + weight_rep_OS['US High Yield']
weights_equities_rep_OS = weight_rep_OS['World Equities']
weights_commodities_rep_OS = weight_rep_OS['Gold'] +  weight_rep_OS['Energy'] + weight_rep_OS['Copper']

#Performance for each type of asset (target)
perf_bonds_target_OS = (np.multiply(returns_OS_price['World Bonds'], weight_target_OS['World Bonds']) + np.multiply(returns_OS_price['US Investment Grade'], weight_target_OS['US Investment Grade']) + np.multiply(returns_OS_price['US High Yield'], weight_target_OS['US High Yield']))/weights_bonds_target_OS
perf_equities_target_OS = (np.multiply(returns_OS_price['World Equities'], weight_target_OS['World Equities']))/weights_equities_target_OS
perf_commodities_target_OS = (np.multiply(returns_OS_price['Gold'], weight_target_OS['Gold']) + np.multiply(returns_OS_price['Energy'], weight_target_OS['Energy']) + np.multiply(returns_OS_price['Copper'], weight_target_OS['Copper']))/weights_commodities_target_OS

#Performance for each type of asset (replicatio/portfolio)
perf_bonds_rep_OS = (np.multiply(returns_OS_price['World Bonds'], weight_rep_OS['World Bonds']) + np.multiply(returns_OS_price['US Investment Grade'], weight_rep_OS['US Investment Grade']) + np.multiply(returns_OS_price['US High Yield'], weight_rep_OS['US High Yield']))/weights_bonds_rep_OS
perf_equities_rep_OS = (np.multiply(returns_OS_price['World Equities'], weight_rep_OS['World Equities']))/weights_equities_rep_OS
perf_commodities_rep_OS = (np.multiply(returns_OS_price['Gold'], weight_rep_OS['Gold']) + np.multiply(returns_OS_price['Energy'], weight_rep_OS['Energy']) + np.multiply(returns_OS_price['Copper'], weight_rep_OS['Copper']))/weights_commodities_rep_OS

R_OS = weights_bonds_rep_OS*perf_bonds_rep_OS +  weights_equities_rep_OS*perf_equities_rep_OS + weights_commodities_rep_OS*perf_commodities_rep_OS

B_OS = weights_bonds_target_OS*perf_bonds_target_OS + weights_equities_target_OS*perf_equities_target_OS + weights_commodities_target_OS*perf_commodities_target_OS

R_S_OS = weights_bonds_target_OS*perf_bonds_rep_OS + weights_equities_target_OS*perf_equities_rep_OS + weights_commodities_target_OS*perf_commodities_rep_OS

B_S_OS = weights_bonds_rep_OS*perf_bonds_target_OS + weights_equities_rep_OS*perf_equities_target_OS + weights_commodities_rep_OS*perf_commodities_target_OS

plt.figure(figsize=(16,7))
plt.subplot(131)
#interaction_OS = R_OS - R_S_OS - B_S_OS + B_S_OS
interaction_OS = R_OS - R_S_OS - B_S_OS + B_OS
plt.plot(interaction_OS, 'r')
plt.title('Monthly Interaction Effect Out-of-Sample')

plt.subplot(132)
#allocation_OS = B_OS - R_S_OS
allocation_OS = B_S_OS - B_OS
plt.plot(allocation_OS, 'b')
plt.title('Monthly Allocation Effect Out-of-Sample')

plt.subplot(133)
#selection_OS = B_OS - B_S_OS
selection_OS = R_S_OS - B_OS
plt.plot(selection_OS, 'g')
plt.title('Monthly Selection Effect Out-of-Sample')
plt.show()
plt.close()

###Collect all annualized Performance Attribution###
performance_attribution = pd.DataFrame({'In-Sample': [np.mean(R_IS, 0)*12, np.mean(B_IS, 0)*12, np.mean(R_S_IS, 0)*12, np.mean(B_S_IS, 0)*12, np.mean(interaction_IS, 0)*12, np.mean(allocation_IS, 0)*12, np.mean(selection_IS, 0)*12],
                                        'Out-of-Sample': [np.mean(R_OS, 0)*12, np.mean(B_OS, 0)*12, np.mean(R_S_OS, 0)*12, np.mean(B_S_OS, 0)*12, np.mean(interaction_OS, 0)*12, np.mean(allocation_OS, 0)*12, np.mean(selection_OS, 0)*12]},
                                       index=['R', 'B', 'R_S', 'B_S', 'Interaction', 'Allocation Effect', 'Selection Effect'])
performance_attribution.to_latex('Output/performance_attribution.tex')
