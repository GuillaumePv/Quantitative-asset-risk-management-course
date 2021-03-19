#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:52:47 2021

@author: guillaume
"""

## Import librairies
#####################
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
import matplotlib.pyplot as plt

os.chdir("/Users/guillaume/MyProjects/PythonProjects/QARM/Assignements/A1")
print("Current working directory: {0}".format(os.getcwd()))

## Importation data
######################
df = pd.read_excel("Data_HEC_QAM_A1.xlsx", engine='openpyxl')

df = df.rename(columns={"Unnamed: 0":"date"})
df['date'] = pd.to_datetime(df['date'],format="%d.%m.%Y")

## Condtion to create the two samples of data
#############################################
in_sample = df.loc[(df['date'] <= pd.to_datetime('2017-12-31'))].iloc[:,1:]
out_sample = df.loc[(df['date'] > pd.to_datetime('2017-12-31'))].iloc[:,1:]  

## Returns of "in-sample" dataset
#################################
num_lines_IS=np.size(in_sample,0)
simpleReturns_IS=((in_sample/in_sample.shift(1))-1).dropna()

## Returns of "out-of-samples" dataset
######################################
num_lines_OS=np.size(out_sample,0)
#simpleReturns_OS=np.divide(out_sample.iloc[2:(num_lines_OS),:],out_sample.iloc[1:(num_lines_OS-1),:])-1
simpleReturns_OS=((out_sample/out_sample.shift(1))-1).dropna()

####################
## Optimizer Part ##
####################

#Optimization function
def SK_criterion(weight,Lambda_RA,Returns_data):
    """
    
    this fucntion computes the expected utility in the MArkowitz case when 
    investors have a preference for skewness and kurtosis 
    Parameters
    ----------
    weight : TYPE
        weights in the investor's portfolio.
    Lambda_RA : TYPE
        the risk aversion aparameter.
    Returns_data : TYPE
        the set of returns.

    Returns
    -------
    criterion : Object
        optimal weights of assets in the portfolio.

    """
    portfolio_return=np.multiply(Returns_data,np.transpose(weight))
    portfolio_return=np.sum(portfolio_return,1)
    mean_ret=np.mean(portfolio_return,0)
    sd_ret=np.std(portfolio_return,0)
    skew_ret=skew(portfolio_return,0)
    kurt_ret=kurtosis(portfolio_return,0)
    W=1
    Wbar=1*(1+0.25/100)
    # use CRRA function + les dérivées => create A,B,C D => permet de créer la taylor expansion
    criterion=np.power(Wbar,1-Lambda_RA)/(1+Lambda_RA)+np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)+Lambda_RA*(Lambda_RA+1)/(6)*np.power(Wbar,-2-Lambda_RA)*np.power(W,3)*skew_ret-Lambda_RA*(Lambda_RA+1)*(Lambda_RA+2)/(24)*np.power(Wbar,-3-Lambda_RA)*np.power(W,4)*kurt_ret
    criterion=-criterion
    return criterion
    
def EV_criterion(weight,Lambda_RA,Returns_data):
    """
    
    this fucntion computes the expected utility in the MArkowitz case when 
    investors have a preference for mean and variance
    Parameters
    ----------
   weight : TYPE
        weights in the investor's portfolio.
    Lambda_RA : TYPE
        the risk aversion aparameter.
    Returns_data : TYPE
        the set of returns.

    Returns
    -------
    criterion : Object 
        optimal weights of assets in the portfolio.

    """
    portfolio_return=np.multiply(Returns_data,np.transpose(weight))
    portfolio_return=np.sum(portfolio_return,1)
    mean_ret=np.mean(portfolio_return,0)
    sd_ret=np.std(portfolio_return,0)
    skew_ret=skew(portfolio_return,0)
    kurt_ret=kurtosis(portfolio_return,0)
    W=1
    Wbar=1*(1+0.25/100)
    criterion=np.power(Wbar,1-Lambda_RA)/(1+Lambda_RA)+np.power(Wbar,-Lambda_RA)*W*mean_ret-Lambda_RA/2*np.power(Wbar,-1-Lambda_RA)*np.power(W,2)*np.power(sd_ret,2)
    #in order to maximize this formula in need to put negative sign => because we use minimze -> -minimzeer = maximizer
    criterion=-criterion
    return criterion

## Function to run the two optimizers (EV and SK) in one time
#############################################################
def Optimizer(returnData):
    """
    

    Parameters
    ----------
    returnData : TYPE
        DESCRIPTION.

    Returns
    -------
    res_SK : Object
        result of mean-variance-skewness-kurtosis optimizer
    res_EV : Object
        result of mean-variance optimizer

    """
    #starting points => 1% in each stock
    # weight for criterion :)
    x0 = np.array([0, 0, 0, 0, 0])+0.01
    
    # constraint for weight
    cons=({'type':'eq', 'fun': lambda x:sum(x)-1})
    Bounds= [(0 , 1) for i in range(0,5)] # boudns of weights -> 0 to 1
    
    Lambda_RA=3 #define teh risk aversion parameter
    
    res_SK = minimize(SK_criterion, x0, method='SLSQP', args=(Lambda_RA,np.array(returnData.iloc[:,0:5])),bounds=Bounds,constraints=cons,options={'disp': True})
    #res_SK.x: give the optimal weight
    res_EV = minimize(EV_criterion, x0, method='SLSQP', args=(Lambda_RA,np.array(returnData.iloc[:,0:5])),bounds=Bounds,constraints=cons,options={'disp': True})
    
    return (res_SK,res_EV)


############
## Part 1 ##
############

opt = Optimizer(simpleReturns_IS)
SK_w = opt[0].x
EV_w = opt[1].x

############
## Part 2 ##
############

## function to  reate the rolling performance for out-sample dataset
#####################################################################
def rollingperf(weight,returns,opt):
    return_test = np.multiply(returns,weight)
    sum_return = np.sum(return_test,1)
    perf = [100]

    for i in range(len(sum_return)):
        value = perf[i]*(1+sum_return.values[i])
        perf.append(value)

    df_perf = pd.DataFrame(perf,columns=["Performance"])
    df_perf.index = [f'week {v}' for v in df_perf.index]
    #à refaire car c'est faut => cum return
    final_perf = sum_return.cumsum()
    if opt == "EV":
        df_perf.plot(title="Optimizer EV")
        plt.savefig('fig/EV.png')
    else:
        df_perf.plot(title="Optimizer SK")
    return final_perf

# EV performance
################
perf_EV_IS = rollingperf(simpleReturns_IS,EV_w,"EV") # pas nécessaire
perf_EV_OS = rollingperf(simpleReturns_OS,EV_w,"EV")

# SK performance
################
perf_SK_IS = rollingperf(simpleReturns_IS,SK_w,"SK") #pas nécessaire
perf_SK_OS = rollingperf(simpleReturns_OS,SK_w,"SK")

############
## Part 3 ##
############

## Function to compute Descriptive Statistics 
#############################################
def Stat_descriptive(data,optimal_w):
    """
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    optimal_w : TYPE
        DESCRIPTION.

    Returns
    -------
    output : Dataframe
        Dataframe of Descriptive Statistics
   

    """

    exp=np.mean(data,0)*52
    vol=np.std(data,0)*np.power(52,0.5)
    skew_ret=skew(data,0)
    kurt_ret=kurtosis(data,0)
    output=pd.DataFrame([optimal_w,exp,vol,skew_ret,kurt_ret],columns=["weight","AVg. Ret.","sigma", "Skew","Kurt"]);
    index = output.index
    columns = output.columns
    output = output.transpose()
    output.columns = columns
    output.index = [i for i in df.iloc[:,1:].columns]
    return output

## Descriptive Statistics with EV optimizer
###########################################
stat_EV_IS = Stat_descriptive(simpleReturns_IS,EV_w)
stat_EV_OS = Stat_descriptive(simpleReturns_OS,EV_w)

## Descriptive Statistics with SK optimizer
###########################################
stat_SK_IS = Stat_descriptive(simpleReturns_IS,SK_w)
stat_SK_OS = Stat_descriptive(simpleReturns_OS,SK_w) 

insam