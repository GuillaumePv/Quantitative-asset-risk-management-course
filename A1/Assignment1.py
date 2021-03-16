#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:15:22 2021

@author: sebastiengorgoni
"""

# =============================================================================
# Libraries
# =============================================================================

import pandas as pd
import numpy as np
import numpy.matlib
import scipy.stats as sc
#import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib

#sns.set_theme(style="darkgrid")

# =============================================================================
# Data Processing
# =============================================================================

# Load data

price_asset = pd.read_excel("Data_HEC_QAM_A1.xlsx") # weekly prices

#Data Info 

price_asset.describe()

#Set the date as an index

price_asset.rename(columns={'Unnamed: 0':'Date'}, inplace=True) #It replace 'Unamed: 0' by 'Date'

price_asset['Date'] = pd.to_datetime(price_asset['Date'])

price_asset.set_index('Date', inplace=True)

#Generate Returns

dfRet = price_asset/price_asset.shift(1) - 1 # weekly prices

#Cumulative Returns Graph

dfCumulRet = (dfRet + 1).cumprod() -1

plt.plot(dfCumulRet['World Equities'], label='World Equities')
plt.plot(dfCumulRet['World Bonds'], label='World Bonds')
plt.plot(dfCumulRet['US High Yield'], label='US High Yield')
plt.plot(dfCumulRet['Oil'], label='Oil')
plt.plot(dfCumulRet['Gold'], label='Gold')
plt.title('Cumulative Return Over Time')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()

#Divide into two sub sample

in_sample = price_asset[price_asset.index <= '2017-12-31']

in_sample_ret = dfRet[dfRet.index<= '2017-12-31']
in_sample_ret.dropna(how='any', inplace=True)
in_sample_ret = pd.DataFrame(in_sample_ret.iloc[:,:])

out_of_sample = price_asset[price_asset.index > '2017-12-31']

out_of_sample_ret = dfRet[dfRet.index > '2017-12-31']
out_of_sample_ret.dropna(how='any', inplace=True)
out_of_sample_ret = pd.DataFrame(out_of_sample_ret.iloc[:,:])

# =============================================================================
# Question 1 
# =============================================================================

# Creation of a function that returns the optimal alloc

def MarkowitzPtf(Expectation,Sigma,LambdaAversion):
   # Computes the markowitz optimal allocation
   e=np.matlib.repmat(1,len(Expectation),1) #generate a column matrix full of 1
   Sigma_inv=np.linalg.inv(Sigma);
   # Computation of the Minimum Variance Portfolio
   MVP1=np.matmul(Sigma_inv,e)
   MVP2=np.matmul(np.transpose(e),MVP1)
   MVP=MVP1/MVP2;
   MVP=pd.DataFrame(MVP)
   # Computation of the weight of the speculative portfolio
   weight=np.matmul(np.matmul(np.transpose(e),Sigma_inv),Expectation)/LambdaAversion;
   # Computation of the speculative portfolio
   Spec_Ptf=np.matmul(Sigma_inv,Expectation)/np.matmul(np.matmul(np.transpose(e),Sigma_inv),Expectation)
   Spec_Ptf=pd.DataFrame(Spec_Ptf)
   final_ptf=weight*Spec_Ptf+(1-weight)*MVP;
   final_ptf=pd.DataFrame(np.transpose(final_ptf))
   return final_ptf

# For loop to compute the optimal portfolio based on rolling 250D data

"""
for i in range(250,len(in_sample_ret.iloc[:,1])):
    #Temporary variable
    temp=in_sample_ret.iloc[(i-250):(i),0:9];
    #Expectation=np.mean(temp, 0)
    #Rolling
    Expectation=np.mean(in_sample_ret.iloc[:,0:9], 0)    
    #Sigma=np.cov(np.transpose(simpleReturns.iloc[:,0:9]));
    Sigma=np.cov(np.transpose(temp));    
    LambdaAversion=3;
    alloc_temp=MarkowitzPtf(Expectation,Sigma,LambdaAversion);
    if i==250:
        optim_alloc_rolling=alloc_temp;
    else:
        optim_alloc_rolling=optim_alloc_rolling.append(alloc_temp);
        
Returns_data=temp;
weight=pd.DataFrame(optim_alloc_rolling.iloc[1,:]);
np.multiply(Returns_data,np.transpose(weight))
"""

#Formula Slide 39, Lecture 2 (only first two moments) 
def EV_criterion(weight,Lambda_RA,Returns_data):
    portfolio_return=np.multiply(Returns_data,np.transpose(weight));
    portfolio_return=np.sum(portfolio_return,1);
    mean_ret=np.mean(portfolio_return,0)
    sd_ret=np.std(portfolio_return,0)
    W=1;
    Wbar=1*(1+0.25/100);
    criterion=np.power(Wbar,1-Lambda_RA)/(1+Lambda_RA)+np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)
    criterion=-criterion;
    return criterion

from scipy.optimize import minimize

x0 = np.array([0, 0, 0, 0, 0])+0.1

cons=({'type':'eq', 'fun': lambda x:sum(x)-1})
Bounds= [(0 , 1) for i in range(0,5)]

Lambda_RA=3

res_EV = minimize(EV_criterion, x0, method='SLSQP', args=(Lambda_RA,np.array(in_sample_ret.iloc[:,0:5])),bounds=Bounds,constraints=cons,options={'disp': True})
 
optimal_weights_EV = res_EV.x

optimal_weights_EV = pd.DataFrame(data=optimal_weights_EV, index=['World Equities', 'World Bonds', 'US High Yield', 'Oil','Gold'], columns=['Weights In-Sample'])

#optimal_weights_in_sample = np.transpose(optimal_weights_in_sample)

# =============================================================================
# Question 2 
# =============================================================================

portfolioRet_EV_outsample = np.matmul(out_of_sample_ret, optimal_weights_EV)

"""
sns.lineplot(x='Date', y=0,
                data=portfolioRet_EV)
plt.show()
"""

ax = plt.gca()
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

plt.plot(portfolioRet_EV_outsample, linewidth=1)
plt.title('Mean-Variance Portfolio Weekly Return (2018-2021)')
plt.show()  

# =============================================================================
# Question 3
# =============================================================================

#Out-of-Sample

exp_EV_outsample = np.mean(portfolioRet_EV_outsample,0)*52
vol_EV_outsample = np.std(portfolioRet_EV_outsample,0)*np.power(52,0.5)
skew_EV_outsample = sc.skew(portfolioRet_EV_outsample,0)
kurt_EV_outsample = sc.kurtosis(portfolioRet_EV_outsample,0)
output_EV_outsample = pd.DataFrame([exp_EV_outsample, vol_EV_outsample, skew_EV_outsample, kurt_EV_outsample]);


#In Sample

portfolioRet_EV_insample = np.matmul(in_sample_ret, optimal_weights_EV)

exp_EV_insample = np.mean(portfolioRet_EV_insample,0)*52
vol_EV_insample = np.std(portfolioRet_EV_insample,0)*np.power(52,0.5)
skew_EV_insample = sc.skew(portfolioRet_EV_insample,0)
kurt_EV_insample = sc.kurtosis(portfolioRet_EV_insample,0)
output_EV_insample = pd.DataFrame([exp_EV_insample, vol_EV_insample, skew_EV_insample, kurt_EV_insample]);


# =============================================================================
# Question 4.1
# =============================================================================

def SK_criterion(weight,Lambda_RA,Returns_data):
    #This function computes the expected utility in the Markowitz case when investors have a preference for skewness and kurtosis
    #Weight: weights in the investor's portfolio
    #Lamba_RA: the risk aversion parameter
    #Returns_data: the set of returns
    
    #Computing the historical performance of the portfolio
    portfolio_return=np.multiply(Returns_data,np.transpose(weight));
    portfolio_return=np.sum(portfolio_return,1);
   
    mean_ret=np.mean(portfolio_return,0)
    sd_ret=np.std(portfolio_return,0)
    skew_ret=sc.skew(portfolio_return,0)
    kurt_ret=sc.kurtosis(portfolio_return,0)
   
    W=1;
    Wbar=1*(1+0.25/100);
   
    A=np.power(Wbar,1-Lambda_RA)/(1+Lambda_RA)
    B=np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)
    C=Lambda_RA*(Lambda_RA+1)/(6)*np.power(Wbar,-2-Lambda_RA)*np.power(W,3)*skew_ret
    D=-Lambda_RA*(Lambda_RA+1)*(Lambda_RA+2)/(24)*np.power(Wbar,-3-Lambda_RA)*np.power(W,4)*kurt_ret
    criterion = A+B+C+D
    #criterion=np.power(Wbar,1-Lambda_RA)/(1+Lambda_RA)+np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)+Lambda_RA*(Lambda_RA+1)/(6)*np.power(Wbar,-2-Lambda_RA)*np.power(W,3)*skew_ret-Lambda_RA*(Lambda_RA+1)*(Lambda_RA+2)/(24)*np.power(Wbar,-3-Lambda_RA)*np.power(W,4)*kurt_ret
    criterion=-criterion;
    return criterion

from scipy.optimize import minimize

x0 = np.array([0, 0, 0, 0, 0])+0.1

cons=({'type':'eq', 'fun': lambda x:sum(x)-1})
Bounds= [(0 , 1) for i in range(0,5)]

Lambda_RA=3

res_SK = minimize(SK_criterion, x0, method='SLSQP', args=(Lambda_RA,np.array(in_sample_ret.iloc[:,0:5])),bounds=Bounds,constraints=cons,options={'disp': True})
 
optimal_weights_SK = res_SK.x

optimal_weights_SK = pd.DataFrame(data=optimal_weights_SK, index=['World Equities', 'World Bonds', 'US High Yield', 'Oil','Gold'], columns=['Weights In-Sample'])

# =============================================================================
# Quesion 4.2
# =============================================================================

portfolioRet_SK_outsample = np.matmul(out_of_sample_ret, optimal_weights_SK)

"""
sns.lineplot(x='Date', y=0,
                data=portfolioRet_EV)
plt.show()
"""

ax = plt.gca()
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

plt.plot(portfolioRet_SK_outsample, linewidth=1)
plt.title('Mean-Variance-Skewness-Kurtosis Portfolio Weekly Return (2018-2021)')
plt.show()

# =============================================================================
# Question 4.3 
# =============================================================================

#Out-of-Sample Portfolio

exp_SK_outsample = np.mean(portfolioRet_SK_outsample,0)*52
vol_SK_outsample = np.std(portfolioRet_SK_outsample,0)*np.power(52,0.5)
skew_SK_outsample = sc.skew(portfolioRet_SK_outsample,0)
kurt_SK_outsample = sc.kurtosis(portfolioRet_SK_outsample,0)
output_SK_outsample = pd.DataFrame([exp_SK_outsample, vol_SK_outsample, skew_SK_outsample, kurt_SK_outsample]);

#In Sample Portfolio

portfolioRet_SK_insample = np.matmul(in_sample_ret, optimal_weights_SK)

exp_SK_insample = np.mean(portfolioRet_SK_insample,0)*52
vol_SK_insample = np.std(portfolioRet_SK_insample,0)*np.power(52,0.5)
skew_SK_insample = sc.skew(portfolioRet_SK_insample,0)
kurt_SK_insample = sc.kurtosis(portfolioRet_SK_insample,0)
output_SK_insample = pd.DataFrame([exp_SK_insample, vol_SK_insample, skew_SK_insample, kurt_SK_insample]);

# =============================================================================
# Extra
# =============================================================================

#Out-of-Sample All Assets

exp_alloutsample=np.mean(out_of_sample_ret,0)*52
vol_alloutsample=np.std(out_of_sample_ret,0)*np.power(52,0.5)
skew_alloutsample=sc.skew(out_of_sample_ret)
kurt_alloutsample=sc.kurtosis(out_of_sample_ret)
output_alloutsample=pd.DataFrame([res_EV.x,res_SK.x, exp_alloutsample, vol_alloutsample, skew_alloutsample, kurt_alloutsample]);

#In-Sample All Assets

exp_allinsample=np.mean(in_sample_ret,0)*52
vol_allinsample=np.std(in_sample_ret,0)*np.power(52,0.5)
skew_allinsample=sc.skew(in_sample_ret)
kurt_allinsample=sc.kurtosis(in_sample_ret)
output_allinsample=pd.DataFrame([res_EV.x,res_SK.x, exp_allinsample, vol_allinsample, skew_allinsample, kurt_allinsample]);
