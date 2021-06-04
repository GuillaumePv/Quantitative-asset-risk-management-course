#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:21:30 2021

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
import random
from tqdm import tqdm
from scipy.stats import norm

random.seed(10)

sns.set(style="whitegrid")

#Set the working directory
#os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Quantitative Asset & Risk Management/Assignments/Assignment 3")
print("Current working directory: {0}".format(os.getcwd()))

from assignment3_part1 import * 
from assignment3_part1 import MCR_calc

#Set the working directory
#os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Quantitative Asset & Risk Management/Assignments/Assignment 3")
#print("Current working directory: {0}".format(os.getcwd()))

#Create files in the working directory
if not os.path.isdir('Plot3'):
    os.makedirs('Plot3')
    
#Create files in the working directory
if not os.path.isdir('Output3'):
    os.makedirs('Output3')

SAA_active_weight_IS = SAA_weights_IS - benchmark_weights_IS 
SAA_active_weight_OS = SAA_weights_OS - benchmark_weights_OS 
SAA_active_weight = pd.concat([SAA_active_weight_IS, SAA_active_weight_OS])

TAA_active_weight_IS = TAA_weights_IS - benchmark_weights_IS 
TAA_active_weight_OS = TAA_weights_OS - benchmark_weights_OS 
TAA_active_weight = pd.concat([TAA_active_weight_IS, TAA_active_weight_OS])

rep_active_weight_IS = weight_rep_IS - benchmark_weights_IS 
rep_active_weight_OS = weight_rep_OS - benchmark_weights_OS 
rep_active_weight = pd.concat([rep_active_weight_IS, rep_active_weight_OS])

# =============================================================================
# =============================================================================
# Question 1
# =============================================================================
# =============================================================================

def MCR(allocation, returns):
    """
    This function computes the marginal contribution to risk (MCR), which 
    determine how much the portfolio volatility would change if we increase
    the weight of a particular asset.    

    Parameters
    ----------
    allocation : TYPE
        Weights in the investor's portfolio
    returns : TYPE
        The returns of the portfolio's assets

    Returns
    -------
    TYPE
        Marginal contribution to risk (MCR)

    """
    alloc = allocation.values
    sigma = returns.cov().values
    num = np.matmul(sigma, alloc)
    temp = np.matmul(alloc.T, sigma)
    var_month = np.matmul(temp, alloc)
    vol_month = np.power(var_month, 0.5)
    return num/vol_month

MCR_SAA = []
for i in range(SAA_active_weight_OS.shape[0]):
    temp = MCR(SAA_active_weight_OS.iloc[i, :], returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :])
    MCR_SAA.append(temp)

exante_MCR_SAA = pd.DataFrame(MCR_SAA, columns=SAA_active_weight_OS.columns, index=SAA_active_weight_OS.index)

MCR_TAA = []
for i in range(TAA_active_weight_OS.shape[0]):
    temp = MCR(TAA_active_weight_OS.iloc[i, :], returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :])
    MCR_TAA.append(temp)

exante_MCR_TAA = pd.DataFrame(MCR_TAA, columns=TAA_active_weight_OS.columns, index=TAA_active_weight_OS.index)

MCR_rep = []
for i in range(rep_active_weight_OS.shape[0]):
    temp = MCR(rep_active_weight_OS.iloc[i, :], returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :])
    MCR_rep.append(temp)

exante_MCR_rep = pd.DataFrame(MCR_rep, columns=rep_active_weight_OS.columns, index=rep_active_weight_OS.index)

exante_MCR_SAA.plot(figsize=(13,8))
plt.title('Ex-Ante MCR of SAA Portfolio')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_MCR.png', bbox_inches='tight')
plt.show()
plt.close()

exante_MCR_TAA.plot(figsize=(13,8))
plt.title('Ex-Ante MCR of TAA Portfolio')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_MCR.png', bbox_inches='tight')
plt.show()
plt.close()

exante_MCR_rep.plot(figsize=(13,8))
plt.title('Ex-Ante MCR of Real Portfolio')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/Real_MCR.png', bbox_inches='tight')
plt.show()
plt.close()

##SAA descriptive stats##
SAA_active_weight_OS.describe().to_latex('Output3/SAA_weights.tex')
exante_MCR_SAA.describe().to_latex('Output3/SAA_MCR.tex')
SAA_OS_RC=SAA_active_weight_OS*exante_MCR_SAA
SAA_OS_RC.describe().to_latex('Output3/SAARC.tex')

##TAA descriptive stats##
TAA_active_weight_OS.describe().to_latex('Output3/TAA_weights.tex')
exante_MCR_TAA.describe().to_latex('Output3/TAA_MCR.tex')
TAA_OS_RC=TAA_active_weight_OS*exante_MCR_TAA
TAA_OS_RC.describe().to_latex('Output3/TAARC.tex')

##Real ptf descriptive stats##
rep_active_weight_OS.describe().to_latex('Output3/Real_weights.tex')
exante_MCR_rep.describe().to_latex('Output3/Real_MCR.tex')
Real_OS_RC=rep_active_weight_OS*exante_MCR_rep
Real_OS_RC.describe().to_latex('Output3/RealRC.tex')

# =============================================================================
# =============================================================================
# Question 2 & 3
# =============================================================================
# =============================================================================

# =============================================================================
# Variance Covariance Method 
# =============================================================================

"""VaR and ES of SAA (Expanding)"""
VaR_95_SAA = []
VaR_90_SAA = []
ES_95_SAA = []
ES_90_SAA = []
for i in tqdm(range(len(SAA_active_weight_OS))):
    #temp = -np.multiply(SAA_active_weight.iloc[:SAA_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(SAA_active_weight.iloc[SAA_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    VaR_95_temp = mean_temp + std_temp*norm.ppf(0.95)
    VaR_90_temp = mean_temp + std_temp*norm.ppf(0.90)
    ES_95_temp = mean_temp + std_temp*((norm.pdf(norm.ppf(0.95)))/0.05)
    ES_90_temp = mean_temp + std_temp*((norm.pdf(norm.ppf(0.90))) / 0.10)
    VaR_95_SAA.append(VaR_95_temp)
    VaR_90_SAA.append(VaR_90_temp)
    ES_95_SAA.append(ES_95_temp)
    ES_90_SAA.append(ES_90_temp)

df_VaR_95_SAA_varcov = pd.DataFrame({'VaR 95%': VaR_95_SAA}, index=SAA_active_weight_OS.index)
df_VaR_90_SAA_varcov = pd.DataFrame({'VaR 90%': VaR_90_SAA}, index=SAA_active_weight_OS.index)
df_ES_95_SAA_varcov = pd.DataFrame({'ES 95%': ES_95_SAA}, index=SAA_active_weight_OS.index)
df_ES_90_SAA_varcov = pd.DataFrame({'ES 90%': ES_90_SAA}, index=SAA_active_weight_OS.index)

SAA_stats_varcov = pd.concat([df_VaR_95_SAA_varcov.describe(), df_VaR_90_SAA_varcov.describe(), df_ES_95_SAA_varcov.describe(), df_ES_90_SAA_varcov.describe()], axis=1)
SAA_stats_varcov.to_latex('Output3/SAA_stats_varcov.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_SAA_varcov, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_SAA_varcov, label='1-Month VaR at 10%')
plt.title('SAA Portfolio using Variance Covariance Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_VaR_varcov.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_SAA_varcov, label='1-Month ES at 5%')
plt.plot(df_ES_90_SAA_varcov, label='1-Month ES at 10%')
plt.title('SAA Portfolio using Variance Covariance Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_ES_varcov.png', bbox_inches='tight')
plt.show()
plt.close()

"""VaR and ES of TAA (Expanding)"""
VaR_95_TAA = []
VaR_90_TAA = []
ES_95_TAA = []
ES_90_TAA = []
for i in tqdm(range(len(TAA_active_weight_OS))):
    #temp = -np.multiply(TAA_active_weight.iloc[:TAA_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(TAA_active_weight.iloc[TAA_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)    
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    VaR_95_temp = mean_temp + std_temp*norm.ppf(0.95)
    VaR_90_temp = mean_temp + std_temp*norm.ppf(0.90)
    ES_95_temp = mean_temp + std_temp*((norm.pdf(norm.ppf(0.95)))/0.05)
    ES_90_temp = mean_temp + std_temp*((norm.pdf(norm.ppf(0.90))) / 0.10)
    VaR_95_TAA.append(VaR_95_temp)
    VaR_90_TAA.append(VaR_90_temp)
    ES_95_TAA.append(ES_95_temp)
    ES_90_TAA.append(ES_90_temp)

df_VaR_95_TAA_varcov = pd.DataFrame({'VaR 95%': VaR_95_TAA}, index=TAA_active_weight_OS.index)
df_VaR_90_TAA_varcov = pd.DataFrame({'VaR 90%': VaR_90_TAA}, index=TAA_active_weight_OS.index)
df_ES_95_TAA_varcov = pd.DataFrame({'ES 95%': ES_95_TAA}, index=TAA_active_weight_OS.index)
df_ES_90_TAA_varcov = pd.DataFrame({'ES 90%': ES_90_TAA}, index=TAA_active_weight_OS.index)

TAA_stats_varcov = pd.concat([df_VaR_95_TAA_varcov.describe(), df_VaR_90_TAA_varcov.describe(), df_ES_95_TAA_varcov.describe(), df_ES_90_TAA_varcov.describe()], axis=1)
TAA_stats_varcov.to_latex('Output3/TAA_stats_varcov.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_TAA_varcov, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_TAA_varcov, label='1-Month VaR at 10%')
plt.title('TAA Portfolio using Variance Covariance Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_VaR_varcov.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_TAA_varcov, label='1-Month ES at 5%')
plt.plot(df_ES_90_TAA_varcov, label='1-Month ES at 10%')
plt.title('TAA Portfolio using Variance Covariance Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_ES_varcov.png', bbox_inches='tight')
plt.show()
plt.close()

"""VaR and ES of Real Ptf (Expanding)"""
VaR_95_Real = []
VaR_90_Real = []
ES_95_Real = []
ES_90_Real = []
for i in tqdm(range(len(rep_active_weight_OS))):
    #temp = -np.multiply(rep_active_weight.iloc[:rep_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(rep_active_weight.iloc[rep_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    VaR_95_temp = mean_temp + std_temp*norm.ppf(0.95)
    VaR_90_temp = mean_temp + std_temp*norm.ppf(0.90)
    ES_95_temp = mean_temp + std_temp*((norm.pdf(norm.ppf(0.95)))/0.05)
    ES_90_temp = mean_temp + std_temp*((norm.pdf(norm.ppf(0.90))) / 0.10)
    VaR_95_Real.append(VaR_95_temp)
    VaR_90_Real.append(VaR_90_temp)
    ES_95_Real.append(ES_95_temp)
    ES_90_Real.append(ES_90_temp)
  
df_VaR_95_Real_varcov = pd.DataFrame({'VaR 95%': VaR_95_Real}, index=rep_active_weight_OS.index)
df_VaR_90_Real_varcov = pd.DataFrame({'VaR 90%': VaR_90_Real}, index=rep_active_weight_OS.index)
df_ES_95_Real_varcov = pd.DataFrame({'ES 95%': ES_95_Real}, index=rep_active_weight_OS.index)
df_ES_90_Real_varcov = pd.DataFrame({'ES 90%': ES_90_Real}, index=rep_active_weight_OS.index)

Real_stats_varcov = pd.concat([df_VaR_95_Real_varcov.describe(), df_VaR_90_Real_varcov.describe(), df_ES_95_Real_varcov.describe(), df_ES_90_Real_varcov.describe()], axis=1)
Real_stats_varcov.to_latex('Output3/Real_stats_varcov.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_Real_varcov, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_Real_varcov, label='1-Month VaR at 10%')
plt.title('Real Portfolio using Variance Covariance Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/rep_VaR_varcov.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_Real_varcov, label='1-Month ES at 5%')
plt.plot(df_ES_90_Real_varcov, label='1-Month ES at 10%')
plt.title('Real Portfolio using Variance Covariance Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/rep_ES_varcov.png', bbox_inches='tight')
plt.show()
plt.close()

# =============================================================================
# Historical Method
# =============================================================================

"""VaR and ES of SAA (Expanding)"""
VaR_95_SAA = []
VaR_90_SAA = []
ES_95_SAA = []
ES_90_SAA = []
for i in tqdm(range(len(SAA_active_weight_OS))):
    #temp = -np.multiply(SAA_active_weight.iloc[:SAA_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(SAA_active_weight.iloc[SAA_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)    
    temp_sort = temp.sort_values(ascending=False)
    VaR_95_temp = temp_sort.quantile(0.95)
    VaR_90_temp = temp_sort.quantile(0.90)
    ES_95_temp = temp[temp > VaR_95_temp].mean()
    ES_90_temp = temp[temp > VaR_90_temp].mean()
    VaR_95_SAA.append(VaR_95_temp)
    VaR_90_SAA.append(VaR_90_temp)
    ES_95_SAA .append(ES_95_temp)
    ES_90_SAA .append(ES_90_temp)

df_VaR_95_SAA_hist = pd.DataFrame({'VaR 95%': VaR_95_SAA}, index=SAA_active_weight_OS.index)
df_VaR_90_SAA_hist = pd.DataFrame({'VaR 90%': VaR_90_SAA}, index=SAA_active_weight_OS.index)
df_ES_95_SAA_hist = pd.DataFrame({'ES 95%': ES_95_SAA}, index=SAA_active_weight_OS.index)
df_ES_90_SAA_hist = pd.DataFrame({'ES 90%': ES_90_SAA}, index=SAA_active_weight_OS.index)

SAA_stats_hist = pd.concat([df_VaR_95_SAA_hist.describe(), df_VaR_90_SAA_hist.describe(), df_ES_95_SAA_hist.describe(), df_ES_90_SAA_hist.describe()], axis=1)
SAA_stats_hist.to_latex('Output3/SAA_stats_hist.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_SAA_hist, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_SAA_hist, label='1-Month VaR at 10%')
plt.title('SAA Portfolio using Historical Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_VaR_HS.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_SAA_hist, label='1-Month ES at 5%')
plt.plot(df_ES_90_SAA_hist, label='1-Month ES at 10%')
plt.title('SAA Portfolio using Historical Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_ES_HS.png', bbox_inches='tight')
plt.show()
plt.close()

"""VaR and ES of TAA (Expanding)"""
VaR_95_TAA = []
VaR_90_TAA = []
ES_95_TAA = []
ES_90_TAA = []
for i in tqdm(range(len(TAA_active_weight_OS))):
    #temp = -np.multiply(TAA_active_weight.iloc[:TAA_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(TAA_active_weight.iloc[TAA_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)    
    temp_sort = temp.sort_values(ascending=False)
    VaR_95_temp = temp_sort.quantile(0.95)
    VaR_90_temp = temp_sort.quantile(0.90)
    ES_95_temp = temp[temp > VaR_95_temp].mean()
    ES_90_temp = temp[temp > VaR_90_temp].mean()
    VaR_95_TAA.append(VaR_95_temp)
    VaR_90_TAA.append(VaR_90_temp)
    ES_95_TAA .append(ES_95_temp)
    ES_90_TAA .append(ES_90_temp)

df_VaR_95_TAA_hist = pd.DataFrame({'VaR 95%': VaR_95_TAA}, index=TAA_active_weight_OS.index)
df_VaR_90_TAA_hist = pd.DataFrame({'VaR 90%': VaR_90_TAA}, index=TAA_active_weight_OS.index)
df_ES_95_TAA_hist = pd.DataFrame({'ES 95%': ES_95_TAA}, index=TAA_active_weight_OS.index)
df_ES_90_TAA_hist = pd.DataFrame({'ES 90%': ES_90_TAA}, index=TAA_active_weight_OS.index)

TAA_stats_hist = pd.concat([df_VaR_95_TAA_hist.describe(), df_VaR_90_TAA_hist.describe(), df_ES_95_TAA_hist.describe(), df_ES_90_TAA_hist.describe()], axis=1)
TAA_stats_hist.to_latex('Output3/TAA_stats_hist.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_TAA_hist, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_TAA_hist, label='1-Month VaR at 10%')
plt.title('TAA Portfolio using Historical Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_VaR_HS.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_TAA_hist, label='1-Month ES at 5%')
plt.plot(df_ES_90_TAA_hist, label='1-Month ES at 10%')
plt.title('TAA Portfolio using Historical Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_ES_HS.png', bbox_inches='tight')
plt.show()
plt.close()

"""VaR and ES of Real Ptf (Expanding)"""
VaR_95_Real = []
VaR_90_Real = []
ES_95_Real = []
ES_90_Real = []
for i in tqdm(range(len(rep_active_weight_OS))):
    #temp = -np.multiply(rep_active_weight.iloc[:rep_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(rep_active_weight.iloc[rep_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp_sort = temp.sort_values(ascending=False)
    VaR_95_temp = temp_sort.quantile(0.95)
    VaR_90_temp = temp_sort.quantile(0.90)
    ES_95_temp = temp[temp > VaR_95_temp].mean()
    ES_90_temp = temp[temp > VaR_90_temp].mean()
    VaR_95_Real.append(VaR_95_temp)
    VaR_90_Real.append(VaR_90_temp)
    ES_95_Real.append(ES_95_temp)
    ES_90_Real.append(ES_90_temp)
    
df_VaR_95_Real_hist = pd.DataFrame({'VaR 95%': VaR_95_Real}, index=rep_active_weight_OS.index)
df_VaR_90_Real_hist = pd.DataFrame({'VaR 90%': VaR_90_Real}, index=rep_active_weight_OS.index)
df_ES_95_Real_hist = pd.DataFrame({'ES 95%': ES_95_Real}, index=rep_active_weight_OS.index)
df_ES_90_Real_hist = pd.DataFrame({'ES 90%': ES_90_Real}, index=rep_active_weight_OS.index)

Real_stats_hist = pd.concat([df_VaR_95_Real_hist.describe(), df_VaR_90_Real_hist.describe(), df_ES_95_Real_hist.describe(), df_ES_90_Real_hist.describe()], axis=1)
Real_stats_hist.to_latex('Output3/Real_stats_hist.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_Real_hist, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_Real_hist, label='1-Month VaR at 10%')
plt.title('Real Portfolio using Historical Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/rep_VaR_HS.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_Real_hist, label='1-Month ES at 5%')
plt.plot(df_ES_90_Real_hist, label='1-Month ES at 10%')
plt.title('Real Portfolio using Historical Method')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/rep_ES_HS.png', bbox_inches='tight')
plt.show()
plt.close()

# =============================================================================
# Age Weighted Simulation
# =============================================================================

#Source: https://rstudio-pubs-static.s3.amazonaws.com/380973_abe685cfa68b4890a82e247c5e8e5869.html#/#https://rstudio-pubs-static.s3.amazonaws.com/380973_abe685cfa68b4890a82e247c5e8e5869.html#/

"""VaR and ES of SAA (Expanding)"""
VaR_95_SAA = []
VaR_90_SAA = []
ES_95_SAA = []
ES_90_SAA = []
for i in tqdm(range(len(SAA_active_weight_OS))):
    #temp = -np.multiply(SAA_active_weight.iloc[:SAA_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(SAA_active_weight.iloc[SAA_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:SAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)    
    lamb = 0.98
    n = len(temp)
    time = [i+1 for i in range(n)]
    time = np.sort(time)[::-1]
    w = []
    for i in range(n):
        weight = ((1-lamb)*(lamb**(time[i]-1)))/(1-lamb**n)
        w.append(weight)
    df_temp = pd.DataFrame({'return': temp, 'w': w})
    df_temp = df_temp.sort_values(by='return',ascending=False)
    df_temp['cumsum_w'] = np.cumsum(df_temp['w'])
    VaR_95_temp = df_temp['return'][df_temp['cumsum_w'] <= 0.05].iloc[-1]
    VaR_90_temp = df_temp['return'][df_temp['cumsum_w'] <= 0.10].iloc[-1]
    ES_95_temp = df_temp['return'][df_temp['return'] > VaR_95_temp].mean()
    ES_90_temp = df_temp['return'][df_temp['return'] > VaR_90_temp].mean()
    VaR_95_SAA.append(VaR_95_temp)
    VaR_90_SAA.append(VaR_90_temp)
    ES_95_SAA .append(ES_95_temp)
    ES_90_SAA .append(ES_90_temp)

plt.figure(figsize=(10,7))
weights = pd.Series(w, index=SAA_active_weight.index)
plt.plot(weights)
plt.title('Evolution of Weight in Age-Weighted Simulatuon')
plt.savefig('Plot3/Weights.png')
plt.show()
plt.close()

df_VaR_95_SAA_age = pd.DataFrame({'VaR 95%': VaR_95_SAA}, index=SAA_active_weight_OS.index)
df_VaR_90_SAA_age = pd.DataFrame({'VaR 90%': VaR_90_SAA}, index=SAA_active_weight_OS.index)
df_ES_95_SAA_age = pd.DataFrame({'ES 95%': ES_95_SAA}, index=SAA_active_weight_OS.index)
df_ES_90_SAA_age = pd.DataFrame({'ES 90%': ES_90_SAA}, index=SAA_active_weight_OS.index)

SAA_stats_age = pd.concat([df_VaR_95_SAA_age.describe(), df_VaR_90_SAA_age.describe(), df_ES_95_SAA_age.describe(), df_ES_90_SAA_age.describe()], axis=1)
SAA_stats_age.to_latex('Output3/SAA_stats_age.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_SAA_age, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_SAA_age, label='1-Month VaR at 10%')
plt.title('SAA Portfolio using Age Weighted Simulation')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_VaR_age.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_SAA_age, label='1-Month ES at 5%')
plt.plot(df_ES_90_SAA_age, label='1-Month ES at 10%')
plt.title('SAA Portfolio using Age Weighted Simulation')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/SAA_ES_age.png', bbox_inches='tight')
plt.show()
plt.close()

"""VaR and ES of TAA (Expanding)"""
VaR_95_TAA = []
VaR_90_TAA = []
ES_95_TAA = []
ES_90_TAA = []
for i in tqdm(range(len(TAA_active_weight_OS))):
    #temp = -np.multiply(TAA_active_weight.iloc[:TAA_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(TAA_active_weight.iloc[TAA_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:TAA_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)    
    lamb = 0.98
    n = len(temp)
    time = [i+1 for i in range(n)]
    time = np.sort(time)[::-1]
    w = []
    for i in range(n):
        weight = ((1-lamb)*(lamb**(time[i]-1)))/(1-lamb**n)
        w.append(weight)
    df_temp = pd.DataFrame({'return': temp, 'w':w})
    df_temp = df_temp.sort_values(by='return',ascending=False)
    df_temp['cumsum_w'] = np.cumsum(df_temp['w'])
    VaR_95_temp = df_temp['return'][df_temp['cumsum_w'] <= 0.05].iloc[-1]
    VaR_90_temp = df_temp['return'][df_temp['cumsum_w'] <= 0.10].iloc[-1]
    ES_95_temp = df_temp['return'][df_temp['return'] > VaR_95_temp].mean()
    ES_90_temp = df_temp['return'][df_temp['return'] > VaR_90_temp].mean()
    VaR_95_TAA.append(VaR_95_temp)
    VaR_90_TAA.append(VaR_90_temp)
    ES_95_TAA .append(ES_95_temp)
    ES_90_TAA .append(ES_90_temp)

df_VaR_95_TAA_age = pd.DataFrame({'VaR 95%': VaR_95_TAA}, index=TAA_active_weight_OS.index)
df_VaR_90_TAA_age = pd.DataFrame({'VaR 90%': VaR_90_TAA}, index=TAA_active_weight_OS.index)
df_ES_95_TAA_age = pd.DataFrame({'ES 95%': ES_95_TAA}, index=TAA_active_weight_OS.index)
df_ES_90_TAA_age = pd.DataFrame({'ES 90%': ES_90_TAA}, index=TAA_active_weight_OS.index)

TAA_stats_age = pd.concat([df_VaR_95_TAA_age.describe(), df_VaR_90_TAA_age.describe(), df_ES_95_TAA_age.describe(), df_ES_90_TAA_age.describe()], axis=1)
TAA_stats_age.to_latex('Output3/TAA_stats_age.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_TAA_age, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_TAA_age, label='1-Month VaR at 10%')
plt.title('TAA Portfolio using Age Weighted Simulation')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_VaR_age.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_TAA_age, label='1-Month ES at 5%')
plt.plot(df_ES_90_TAA_age, label='1-Month ES at 10%')
plt.title('TAA Portfolio using Age Weighted Simulation')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/TAA_ES_age.png', bbox_inches='tight')
plt.show()
plt.close()

"""VaR and ES of Real Ptf (Expanding)"""
VaR_95_Real = []
VaR_90_Real = []
ES_95_Real = []
ES_90_Real = []
for i in tqdm(range(len(rep_active_weight_OS))):
    #temp = -np.multiply(rep_active_weight.iloc[:rep_active_weight_IS.shape[0]+1+i, :], returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    temp = -(rep_active_weight.iloc[rep_active_weight_IS.shape[0]+i, :]*returns_price.iloc[:rep_active_weight_IS.shape[0]+1+i, :]).sum(axis=1)
    lamb = 0.98
    n = len(temp)
    time = [i+1 for i in range(n)]
    time = np.sort(time)[::-1]
    w = []
    for i in range(n):
        weight = ((1-lamb)*(lamb**(time[i]-1)))/(1-lamb**n)
        w.append(weight)
    df_temp = pd.DataFrame({'return': temp, 'w':w})
    df_temp = df_temp.sort_values(by='return',ascending=False)
    df_temp['cumsum_w'] = np.cumsum(df_temp['w'])
    VaR_95_temp = df_temp['return'][df_temp['cumsum_w'] <= 0.05].iloc[-1]
    VaR_90_temp = df_temp['return'][df_temp['cumsum_w'] <= 0.10].iloc[-1]
    ES_95_temp = df_temp['return'][df_temp['return'] > VaR_95_temp].mean()
    ES_90_temp = df_temp['return'][df_temp['return'] > VaR_90_temp].mean()
    VaR_95_Real.append(VaR_95_temp)
    VaR_90_Real.append(VaR_90_temp)
    ES_95_Real.append(ES_95_temp)
    ES_90_Real.append(ES_90_temp)
    
df_VaR_95_Real_age = pd.DataFrame({'VaR 95%': VaR_95_Real}, index=rep_active_weight_OS.index)
df_VaR_90_Real_age = pd.DataFrame({'VaR 90%': VaR_90_Real}, index=rep_active_weight_OS.index)
df_ES_95_Real_age = pd.DataFrame({'ES 95%': ES_95_Real}, index=rep_active_weight_OS.index)
df_ES_90_Real_age = pd.DataFrame({'ES 90%': ES_90_Real}, index=rep_active_weight_OS.index)

Real_stats_age = pd.concat([df_VaR_95_Real_age.describe(), df_VaR_90_Real_age.describe(), df_ES_95_Real_age.describe(), df_ES_90_Real_age.describe()], axis=1)
Real_stats_age.to_latex('Output3/Real_stats_age.tex')

plt.figure(figsize=(13,8))
plt.plot(df_VaR_95_Real_age, label='1-Month VaR at 5%')
plt.plot(df_VaR_90_Real_age, label='1-Month VaR at 10%')
plt.title('Real Portfolio using Age Weighted Simulation')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/Real_VaR_age.png', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(13,8))
plt.plot(df_ES_95_Real_age, label='1-Month ES at 5%')
plt.plot(df_ES_90_Real_age, label='1-Month ES at 10%')
plt.title('Real Portfolio using Age Weighted Simulation')
plt.ylabel('Monthly Loss')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot3/Real_ES_age.png', bbox_inches='tight')
plt.show()
plt.close()

