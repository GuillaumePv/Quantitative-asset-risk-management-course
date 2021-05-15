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

#sns.set_theme(style="whitegrid")
#
#os.chdir("/Users/guillaume/MyProjects/PythonProjects/QARM/assignements/A2/")
print("Current working directory: {0}".format(os.getcwd()))

df_price = pd.read_excel('Data_QAM2.xlsx', 'Prices', engine='openpyxl')
df_price.set_index('Dates', inplace=True)

benchmark = pd.DataFrame(data=(0.5*df_price['World Equities'] + 0.5*df_price['World Bonds']))

in_sample_price = df_price.loc[(df_price.index <= pd.to_datetime('2010-12-31'))].iloc[:,:]
out_sample_price = df_price.loc[(df_price.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

returns_price = np.log(df_price/df_price.shift(1)).replace(np.nan, 0)
returns_IS_price = np.log(in_sample_price/in_sample_price.shift(1)).replace(np.nan, 0)

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

labels=list(returns_IS_price);

#Plot the optimal weights using ERC
weight_to_chart=np.array(res_ERC.x)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation ERC')
plt.ylabel('Ptf Weight')
plt.tight_layout()

#MCR for Max SR
MCR_chart=MCR_calc(res_ERC.x, returns_IS_price);
MCR_chart=np.array(MCR_chart);
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR ERC')
plt.ylabel('MCR')
plt.tight_layout()

#Test for the MCR/Risk ratio
ratio=res_ERC.x*MCR_chart
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('alpha x MCR')
plt.ylabel('Ratio')
plt.tight_layout()

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

#Plot the optimal weights using Sharpe Ratio
weight_to_chart=np.array(res_SR.x)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation SR')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_SR.x, returns_IS_price);
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

weight_to_chart=np.array(res_MDP.x)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation Max Div')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_MDP.x, returns_IS_price);
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
portfolio_IS_ERC = pd.DataFrame({'ERC': np.sum(np.multiply(returns_IS_price,np.transpose(res_ERC.x)), 1)})
portfolio_IS_SR = pd.DataFrame({'SR': np.sum(np.multiply(returns_IS_price,np.transpose(res_SR.x)), 1)})
portfolio_IS_MDP = pd.DataFrame({'MDP': np.sum(np.multiply(returns_IS_price,np.transpose(res_MDP.x)), 1)})

def hit_ratio(return_dataset):
    return len(return_dataset[return_dataset > 0]) / len(return_dataset)

def max_drawdown(cum_returns):
    roll_max = cum_returns.cummax()
    monthly_drawdown = cum_returns/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    return max_monthly_drawdown
    
max_drawdown((portfolio_IS_ERC['ERC']+1).cumprod()).min()

def perf(data, benchmark, name):
    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.plot(data, 'b')
    plt.title("Monthly Returns", fontsize=15)
    exp = np.mean(data,0)*12
    vol = np.std(data,0)*np.power(12,0.5)
    sharpe = exp/vol
    max_dd = max_drawdown((data+1).cumprod())
    plt.subplot(122)
    plt.plot(max_dd, 'g')
    plt.title("Evolution of Max Drawdown", fontsize=15)
    plt.show()
    plt.close()
    hit = hit_ratio(data)
    df = pd.DataFrame({name: [exp, vol, sharpe, max_dd.min(), hit]}, index = ['Mean', 'Volatility', 'SharpeRatio', 'MaxDrawdown', 'HitRatio'])
    plt.plot((data + 1).cumprod()*100, 'b', label='Portfolio')
    plt.plot((benchmark + 1).cumprod()*100, 'r', label='Benchmark')
    plt.title("Cumulative Returns vs Benchmark", fontsize=15)
    plt.legend(loc='upper left', frameon=True)
    plt.show()
    plt.close()
    return df

ERC_IS_results = perf(portfolio_IS_ERC['ERC'], benchmark_return_IS['IS Benchmark'], 'ERC')
SR_IS_results = perf(portfolio_IS_SR['SR'], benchmark_return_IS['IS Benchmark'], 'SR')
MDP_IS_results = perf(portfolio_IS_MDP['MDP'], benchmark_return_IS['IS Benchmark'], 'MDP')
SAA_IS_results = pd.concat([ERC_IS_results, SR_IS_results, MDP_IS_results], axis=1)

# =============================================================================
# 1.2
# =============================================================================

portfolio_OS_ERC = pd.DataFrame({'ERC': np.sum(np.multiply(returns_OS_price,np.transpose(res_ERC.x)), 1)})
portfolio_OS_SR = pd.DataFrame({'SR': np.sum(np.multiply(returns_OS_price,np.transpose(res_SR.x)), 1)})
portfolio_OS_MDP = pd.DataFrame({'MDP': np.sum(np.multiply(returns_OS_price,np.transpose(res_MDP.x)), 1)})

ERC_OS_results = perf(portfolio_OS_ERC['ERC'], benchmark_return_OS['OS Benchmark'], 'ERC')
SR_OS_results = perf(portfolio_OS_SR['SR'], benchmark_return_OS['OS Benchmark'], 'SR')
MDP_OS_results = perf(portfolio_OS_MDP['MDP'], benchmark_return_OS['OS Benchmark'], 'MDP')
SAA_OS_results = pd.concat([ERC_OS_results, SR_OS_results, MDP_OS_results], axis=1)

# =============================================================================
# =============================================================================
# Part 2
# =============================================================================
# =============================================================================

# =============================================================================
# 2.1
# =============================================================================

df_carry = pd.read_excel("Data_QAM2.xlsx",sheet_name='Carry',engine='openpyxl')
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

returns_IS_quantile = returns_IS_past11.T.apply(lambda x: pd.qcut(x, 5, labels=False), axis=0).T

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
value_IS_results = perf(value_IS_scaled['Value'], benchmark_return_IS['IS Benchmark'], 'Value')
mom_IS_results = perf(mom_IS_scaled['Momentum'], benchmark_return_IS['IS Benchmark'], 'Momentum')

TAA_IS_scaled = pd.DataFrame({'TAA_IS': value_IS_scaled['Value'] + mom_IS_scaled['Momentum']}).replace(np.nan, 0)

TAA_IS_results = pd.concat([value_IS_results, mom_IS_results], axis=1)
TAA_IS_results['Total'] = perf(TAA_IS_scaled['TAA_IS'], benchmark_return_IS['IS Benchmark'], 'TAA_IS')

# =============================================================================
# 2.2
# =============================================================================

df_vix = pd.read_excel("Data_QAM2.xlsx",sheet_name='VIX',engine='openpyxl')
df_vix.set_index('Dates', inplace=True)

df_vix = (df_vix - df_vix.mean())/df_vix.std()
df_vix['Percentage Change'] = np.log(df_vix['VIX']/df_vix['VIX'].shift(1))
df_vix['Percentage Change'] = df_vix['Percentage Change'].replace(np.nan, 0)

plt.plot(df_vix['Percentage Change'])

df_vix_IS = df_vix.loc[(df_vix.index <= pd.to_datetime('2010-12-31'))]
plt.plot(df_vix_IS['VIX'])
plt.plot(df_vix_IS['Percentage Change'])
df_vix_OS = df_vix.loc[(df_vix.index > pd.to_datetime('2010-12-31'))]

corr_value_IS = sns.heatmap(pd.concat([df_vix_IS['VIX'], value_IS_scaled], axis=1).corr(), annot=True)
corr_mom_IS = sns.heatmap(pd.concat([df_vix_IS['VIX'], mom_IS_scaled], axis=1).corr(), annot=True)

plt.plot(df_vix_IS['VIX'].rolling(10).corr(value_IS_scaled))
plt.plot(df_vix_IS['VIX'].rolling(10).corr(mom_IS_scaled))

quantile = df_vix_IS.quantile(q=0.90)
df_vix_IS['Quantile'] = np.ones(df_vix_IS.shape[0])*quantile.VIX

plt.plot(df_vix_IS['VIX'], label = 'Standardized VIX')
plt.plot(df_vix_IS['Quantile'], label = '90% Quantile')
plt.legend(loc='upper left', frameon=True)

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

TAA_IS_VIX_results = perf(TAA_IS_VIX['Returns Strategy'], benchmark_return_IS['IS Benchmark'], 'TAA_IS_VIX')


###Complex Strategy###

df_vix_IS.loc[(df_vix_IS['VIX'] <= quantile.VIX), 'Value Position'] = 0.5 #Long Value
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX)  & (df_vix_IS['Percentage Change'] >= 0), 'Value Position'] = 0 #Short Value
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX)  & (df_vix_IS['Percentage Change'] < 0), 'Value Position'] = 2 #Long Value

df_vix_IS.loc[df_vix_IS['VIX'] <= quantile.VIX, 'Mom Position'] = 0.5 #Long Mom
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX)  & (df_vix_IS['Percentage Change'] >= 0), 'Mom Position'] = 1 #Long Value
df_vix_IS.loc[(df_vix_IS['VIX'] > quantile.VIX) & (df_vix_IS['Percentage Change'] < 0), 'Mom Position'] = -1 #Long Value

TAA_IS_VIX = pd.DataFrame({'Returns Strategy': value_IS_scaled['Value']*df_vix_IS['Value Position'] + mom_IS_scaled['Momentum']*df_vix_IS['Mom Position']}).replace(np.nan, 0)

TAA_IS_VIX_results = perf(TAA_IS_VIX['Returns Strategy'], benchmark_return_IS['IS Benchmark'], 'TAA_IS_VIX')

###Information Ratio###

#https://www.jstor.org/stable/4480091?seq=1#metadata_info_tab_contents
#https://en.wikipedia.org/wiki/Information_ratio

ER_IS = TAA_IS_VIX['Returns Strategy'] - benchmark_return_IS['IS Benchmark']

IR_IS = (np.mean(ER_IS, 0)*12)/np.std(ER_IS, 0)*np.power(12,0.5)

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

returns_quantile = returns_past11.T.apply(lambda x: pd.qcut(x, 5, labels=False,duplicates='drop'), axis=0).T

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
value_OS_results = perf(value_OS_scaled['Value'], benchmark_return_OS['OS Benchmark'], 'Value')
mom_OS_results = perf(mom_OS_scaled['Momentum'], benchmark_return_OS['OS Benchmark'], 'Momentum')

###Out-Sample VIX###
plt.plot(df_vix_OS['VIX'].rolling(10).corr(value_OS_scaled))
plt.plot(df_vix_OS['VIX'].rolling(10).corr(mom_OS_scaled))

quantile = df_vix_IS.quantile(q=0.95)
df_vix_OS['Quantile'] = np.ones(df_vix_OS.shape[0])*quantile.VIX

plt.plot(df_vix_OS['VIX'], label = 'Standardized VIX')
plt.plot(df_vix_OS['Quantile'], label = '90% Quantile In-Sample')
plt.legend(loc='upper left', frameon=True)

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
TAA_OS_VIX_results = perf(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'], 'TAA_OS_VIX')

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

TAA_OS_VIX_results = perf(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'], 'TAA_OS_VIX')

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

SAA_ERC_weights_IS = pd.DataFrame({'World Equities': np.ones(returns_IS_price.shape[0])*res_ERC.x[0],
                                   'World Bonds': np.ones(returns_IS_price.shape[0])*res_ERC.x[1],
                                   'US Investment Grade': np.ones(returns_IS_price.shape[0])*res_ERC.x[2],
                                   'US High Yield': np.ones(returns_IS_price.shape[0])*res_ERC.x[3],
                                   'Gold': np.ones(returns_IS_price.shape[0])*res_ERC.x[4],
                                   'Energy': np.ones(returns_IS_price.shape[0])*res_ERC.x[5],
                                   'Copper': np.ones(returns_IS_price.shape[0])*res_ERC.x[6]
                                   }, index = returns_IS_price.index)

SAA_ERC_weights_OS = pd.DataFrame({'World Equities': np.ones(returns_OS_price.shape[0])*res_ERC.x[0],
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

ptf_target_IS = SAA_ERC_weights_IS + TAA_weights_IS

ptf_target_OS = SAA_ERC_weights_OS + TAA_weights_OS

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

###Ex Ante Tracking Error & Replication Portfolio###
cons = ({'type':'eq', 'fun': lambda weight_ptf: sum(weight_ptf)-1})

Bounds = [(-10 , 10), (-10 , 10), (0 , 0), (-10 , 10), (-10 , 10), (-10 , 10), (-10 , 10)]

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

weight_opt_IS = []
TE_IS = []
for i in range(0, ptf_target_IS.shape[0]):
    x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001
    weight_target = ptf_target_IS.iloc[i, :].values
    sigma = sigma_IS
    opt_TE_IS = minimize(TE, x0, method='Powell', bounds=Bounds, constraints=cons, options={'disp': True}) #Powell works
    weight_opt_IS.append(opt_TE_IS.x)
    TE_IS.append(opt_TE_IS.fun)
 
TE_IS = pd.DataFrame({'TE_IS': TE_IS}, index = ptf_target_IS.index)
weight_rep_IS = np.array(weight_opt_IS)
weight_rep_IS = pd.DataFrame({'World Equities': weight_rep_IS[:, 0],
                              'World Bonds': weight_rep_IS[:, 1],
                              'US Investment Grade': weight_rep_IS[:, 2],
                              'US High Yield': weight_rep_IS[:, 3],
                              'Gold': weight_rep_IS[:, 4],
                              'Energy': weight_rep_IS[:, 5],
                              'Copper': weight_rep_IS[:, 6]}, index = ptf_target_IS.index)

plt.plot(TE_IS)
plt.title('Ex-Ante Tracking Error between Real Ptf vs. SAA+TAA')
plt.show()

return_target_IS  = np.sum(np.multiply(returns_IS_price, ptf_target_IS), axis=1)
return_rep_IS  = np.sum(np.multiply(returns_IS_price, weight_rep_IS), axis=1)

perf_target_IS = perf(return_target_IS, benchmark_return_IS['IS Benchmark'], 'Model')
perf_rep_IS = perf(return_rep_IS, benchmark_return_IS['IS Benchmark'], 'Replication')

plt.plot((return_target_IS+1).cumprod()*100, 'b', label='Model Portfolio')
plt.plot((return_rep_IS+1).cumprod()*100, 'r', label='Replication Portfolio')
plt.title('In-Sample Cumulative Returns')
plt.legend(loc='upper left', frameon=True)
plt.show()

###Ex Post Tracking Error & Replication Portfolio###
cons = ({'type':'eq', 'fun': lambda x: sum(x)-1})

Bounds = [(-10 , 10), (-10 , 10), (0 , 0), (-10 , 10), (-10 , 10), (-10 , 10), (-10 , 10)]

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001

weight_opt_OS = []
TE_OS = []
for i in range(0, ptf_target_OS.shape[0]):
    x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001
    weight_target = ptf_target_OS.iloc[i, :].values
    sigma = sigma_OS
    opt_TE_OS = minimize(TE, x0, method='Powell', bounds=Bounds, constraints=cons, options={'disp': True}) #Powell works
    weight_opt_OS.append(opt_TE_OS.x)
    TE_OS.append(opt_TE_OS.fun)
 
TE_OS = pd.DataFrame({'TE_OS': TE_OS}, index = ptf_target_OS.index)
weight_rep_OS = np.array(weight_opt_OS)
weight_rep_OS = pd.DataFrame({'World Equities': weight_rep_OS[:, 0],
                              'World Bonds': weight_rep_OS[:, 1],
                              'US Investment Grade': weight_rep_OS[:, 2],
                              'US High Yield': weight_rep_OS[:, 3],
                              'Gold': weight_rep_OS[:, 4],
                              'Energy': weight_rep_OS[:, 5],
                              'Copper': weight_rep_OS[:, 6]}, index = ptf_target_OS.index)

plt.plot(TE_OS)
plt.title('Ex-Post Tracking Error')
plt.show()

return_target_OS  = np.sum(np.multiply(returns_OS_price, ptf_target_OS), axis=1)
return_rep_OS  = np.sum(np.multiply(returns_OS_price, weight_rep_OS), axis=1)

perf_target_OS = perf(return_target_OS, benchmark_return_OS['OS Benchmark'], 'Model')
perf_rep_OS = perf(return_rep_OS, benchmark_return_OS['OS Benchmark'], 'Replication')

plt.plot((return_target_OS+1).cumprod()*100, 'b', label='Model Portfolio')
plt.plot((return_rep_OS+1).cumprod()*100, 'r', label='Replication Portfolio')
plt.title('Out-of-Sample Cumulative Returns')
plt.legend(loc='upper left', frameon=True)
plt.show()

###Ex Ante Tracking Error between Target Ptf and SAA###

Tot_TE_SAAvsTarget_IS = []
for i in range(0, ptf_target_IS.shape[0]):
    weight_target = ptf_target_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTarget_IS = TE(SAA_ERC_weights_IS.iloc[i, :].values)
    Tot_TE_SAAvsTarget_IS.append(TE_SAAvsTarget_IS)
 
TE_SAAvsTarget_IS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsTarget_IS}, index = ptf_target_IS.index)

plt.plot(TE_SAAvsTarget_IS)
plt.title('Ex-Ante Tracking Error between SAA vs. SAA+TAA')
plt.show()

###Ex Post Tracking Error between Target Ptf and SAA###

Tot_TE_SAAvsTarget_OS = []
for i in range(0, ptf_target_OS.shape[0]):
    weight_target = ptf_target_OS.iloc[i, :].values
    sigma = sigma_OS
    TE_SAAvsTarget_OS = TE(SAA_ERC_weights_OS.iloc[i, :].values)
    Tot_TE_SAAvsTarget_OS.append(TE_SAAvsTarget_OS)
 
TE_SAAvsTarget_OS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsTarget_OS}, index = ptf_target_OS.index)

plt.plot(TE_SAAvsTarget_OS)
plt.title('Ex-Post Tracking Error between SAA vs. SAA+TAA')
plt.show()
