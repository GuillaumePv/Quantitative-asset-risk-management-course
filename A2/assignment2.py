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

#os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Quantitative Asset & Risk Management/Assignments/Assignment 2")
print("Current working directory: {0}".format(os.getcwd()))

df_price = pd.read_excel('Data_QAM2.xlsx', 'Prices')
df_price.set_index('Dates', inplace=True)

benchmark = pd.DataFrame(data=(0.5*df_price['World Equities'] + 0.5*df_price['World Bonds']))

in_sample_price = df_price.loc[(df_price.index < pd.to_datetime('2010-12-31'))].iloc[:,:]
out_sample_price = df_price.loc[(df_price.index >= pd.to_datetime('2010-12-31'))].iloc[:,:]

in_benchmark = pd.DataFrame(0.5*in_sample_price['World Equities'] + 0.5*in_sample_price['World Bonds'])
os_benchmark = pd.DataFrame(0.5*out_sample_price['World Equities'] + 0.5*out_sample_price['World Bonds'])

returns_price = ((df_price/df_price.shift(1))-1).dropna()
returns_IS_price = ((in_sample_price/in_sample_price.shift(1))-1).dropna()
returns_OS_price =((out_sample_price/out_sample_price.shift(1))-1).dropna()

benchmark_return_IS = ((in_benchmark/in_benchmark.shift(1))-1).dropna()
benchmark_return_OS = ((os_benchmark/os_benchmark.shift(1))-1).dropna()

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
      {'type':'ineq', 'fun': lambda x: x[1]},
      {'type':'ineq', 'fun': lambda x: x[2]})
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

#MCR for Max SR => Beug
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

cons=({'type':'eq', 'fun': lambda x: sum(x)-1},
      {'type':'ineq', 'fun': lambda x: x[1]},
      {'type':'ineq', 'fun': lambda x: x[2]})
Bounds= [(0 , 1) for i in range(0,7)]
#Bounds = [(0 , 1), (0.1 , 1), (0.1 , 1), (0 , 1), (0 , 1), (0 , 1), (0 , 1)]

res_SR = minimize(Max_SR, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

#problème ici pour les world equities
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

cons=({'type':'eq', 'fun': lambda x:sum(x)-1})
#Bounds= [(0 , 1) for i in range(0, 7)]
Bounds = [(0 , 1), (0.1 , 1), (0.1 , 1), (0 , 1), (0 , 1), (0 , 1), (0 , 1)]

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

portfolio_IS_ERC = np.sum(np.multiply(returns_IS_price,np.transpose(res_ERC.x)), 1)
portfolio_IS_SR = np.sum(np.multiply(returns_IS_price,np.transpose(res_SR.x)), 1)
portfolio_IS_MDP = np.sum(np.multiply(returns_IS_price,np.transpose(res_MDP.x)), 1)

def perf(data):
    exp = np.mean(data,0)*12
    vol = np.std(data,0)*np.power(12,0.5)
    sharpe = exp/vol
    roll_max = data.cummax()
    draw = data/roll_max - 1.0
    max_drawdown = np.min(draw.cummin())
    return (exp, vol, sharpe, max_drawdown)

perf(portfolio_IS_ERC)
perf(portfolio_IS_SR)
perf(portfolio_IS_MDP)

plt.plot((portfolio_IS_ERC+1).cumprod()*100)
plt.plot((benchmark_return_IS+1).cumprod()*100)

plt.plot((benchmark_return_OS+1).cumprod()*100)

# =============================================================================
# 1.2
# =============================================================================

portfolio_OS_ERC = np.sum(np.multiply(returns_OS_price,np.transpose(res_ERC.x)), 1)
portfolio_OS_SR = np.sum(np.multiply(returns_OS_price,np.transpose(res_SR.x)), 1)
portfolio_OS_MDP = np.sum(np.multiply(returns_OS_price,np.transpose(res_MDP.x)), 1)

perf(portfolio_OS_ERC)
perf(portfolio_OS_SR)
perf(portfolio_OS_MDP)

# =============================================================================
# =============================================================================
# Part 2
# =============================================================================
# =============================================================================

df_carry = pd.read_excel('Data_QAM2.xlsx', 'Carry')
df_carry.rename(columns={"Unnamed: 0":"Dates"}, inplace=True)
df_carry.set_index('Dates', inplace=True)
