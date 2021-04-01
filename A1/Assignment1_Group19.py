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
from matplotlib import style
import seaborn as sns

style.use('fivethirtyeight') # librairies stylé

os.chdir("/Users/guillaume/MyProjects/PythonProjects/QARM/Assignements/A1")
print("Current working directory: {0}".format(os.getcwd()))

## Importation data
######################
df = pd.read_excel("Data_HEC_QAM_A1.xlsx", engine='openpyxl')

df = df.rename(columns={"Unnamed: 0":"date"})
df['date'] = pd.to_datetime(df['date'],format="%d.%m.%Y")

df.index = df['date']

## Condtion to create the two samples of data
#############################################
in_sample = df.loc[(df['date'] <= pd.to_datetime('2017-12-31'))].iloc[:,1:]
corr_insample = sns.heatmap(in_sample.corr(),annot=True) #Corrrelation matrix between assets (In-sample)
#corr_insample.set_title("Correlation matrix between assets (In-sample dataset)",pad=20,fontweight='bold')
plt.savefig("fig/corr_insample.png", bbox_inches = "tight")

out_sample = df.loc[(df['date'] > pd.to_datetime('2017-12-31'))].iloc[:,1:]  
corr_outsample = sns.heatmap(out_sample.corr(),annot=True) #Correlation matrix between assets (Out-sample)
#corr_outsample.set_title("Correlation matrix between assets (Out-sample dataset)",pad=20,fontweight='bold')
plt.savefig("fig/corr_outsample.png", bbox_inches = "tight")

## Returns of "in-sample" dataset
#################################
num_lines_IS=np.size(in_sample,0)
simpleReturns_IS=((in_sample/in_sample.shift(1))-1).dropna()

## Returns of "out-of-samples" dataset
######################################
num_lines_OS=np.size(out_sample,0)
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
    df_perf.index = out_sample.index
    #à refaire car c'est faut => cum return
    final_perf = sum_return.cumsum()
    if opt == "EV":
        df_perf.plot()
        plt.ylabel('Performance')
        plt.xlabel('Date')
        # plt.title(label="Cumul Perf. of Mean-Var portfolio by indexing at 100 in january 2018", 
        #   pad=20,
        #   fontweight='bold', 
        #   color="black")
        plt.savefig('fig/EV_outsample.png', bbox_inches = "tight")
    else:
        df_perf.plot()
        plt.ylabel('Performance')
        plt.xlabel('Date')
        # plt.title(label="Cumul Perf. of Mean-Var-Skew-Kurt portfolio by indexing at 100 in january 2018", 
        #   pad=20,
        #   fontweight='bold',
        #   color="black")
        plt.savefig('fig/SK_outsample.png', bbox_inches = "tight")
    return final_perf

# EV performance
################
#perf_EV_IS = rollingperf(simpleReturns_IS,EV_w,"EV") # Not necessary
perf_EV_OS = rollingperf(simpleReturns_OS,EV_w,"EV")

# SK performance
################
#perf_SK_IS = rollingperf(simpleReturns_IS,SK_w,"SK") # Not necessary
perf_SK_OS = rollingperf(simpleReturns_OS,SK_w,"SK")

############
## Part 3 ##
############

## Function to compute Descriptive Statistics 
#############################################
def Stat_descriptive(data,optimal_w,opt,sample):
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
    if opt == "EV":
        if sample == "IS":
            output.to_latex("table/IS_EV_stats_decript.tex")
        else:
            output.to_latex("table/OS_EV_stats_decript.tex")
    else:
        if sample == "IS":
            output.to_latex("table/IS_SK_stats_decript.tex")
        else:
            output.to_latex("table/OS_SK_stats_decript.tex")
    return output

## Descriptive Statistics with EV optimizer
###########################################
stat_EV_IS = Stat_descriptive(simpleReturns_IS,EV_w,"EV","IS")
stat_EV_OS = Stat_descriptive(simpleReturns_OS,EV_w,"EV","OS")

## Descriptive Statistics with SK optimizer
###########################################
stat_SK_IS = Stat_descriptive(simpleReturns_IS,SK_w,"SK","IS")
stat_SK_OS = Stat_descriptive(simpleReturns_OS,SK_w,"SK","OS") 

## Compute STD of optimal portfolio ##

#EV

port_return_IS_EV=np.multiply(simpleReturns_IS,np.transpose(EV_w))
port_return_IS_EV=np.sum(port_return_IS_EV,1)
exp_IS_EV=np.mean(port_return_IS_EV,0)*52
sd_re_IS_EV=np.std(port_return_IS_EV,0)*np.power(52,0.5)
skew_ret_IS_EV=skew(port_return_IS_EV,0)
kurt_ret_IS_EV=kurtosis(port_return_IS_EV,0)

port_return_OS_EV=np.multiply(simpleReturns_OS,np.transpose(EV_w))
port_return_OS_EV=np.sum(port_return_OS_EV,1)
exp_OS_EV=np.mean(port_return_OS_EV,0)*52
sd_re_OS_EV=np.std(port_return_OS_EV,0)*np.power(52,0.5)
skew_ret_OS_EV=skew(port_return_OS_EV,0)
kurt_ret_OS_EV=kurtosis(port_return_OS_EV,0)

sd_data_EV = {"Annualized return":[exp_IS_EV,exp_OS_EV],"Volatility":[sd_re_IS_EV,sd_re_OS_EV],"Skewness":[skew_ret_IS_EV,skew_ret_OS_EV],"Kurtosis":[kurt_ret_IS_EV,kurt_ret_OS_EV]}
sd_EV = pd.DataFrame(sd_data_EV,index=['Mean-Variance Portfolio','Skew-Kurtosis Portfolio'])
sd_EV.to_latex("table/comparison_EV.tex")

#SK

port_return_IS_SK=np.multiply(simpleReturns_IS,np.transpose(SK_w))
port_return_IS_SK=np.sum(port_return_IS_SK,1)
exp_IS_SK=np.mean(port_return_IS_SK,0)*52
sd_re_IS_SK=np.std(port_return_IS_SK,0)*np.power(52,0.5)
skew_ret_IS_SK=skew(port_return_IS_SK,0)
kurt_ret_IS_SK=kurtosis(port_return_IS_SK,0)

port_return_OS_SK=np.multiply(simpleReturns_OS,np.transpose(SK_w))
port_return_OS_SK=np.sum(port_return_OS_SK,1)
exp_OS_SK=np.mean(port_return_OS_SK,0)*52
sd_re_OS_SK=np.std(port_return_OS_SK,0)*np.power(52,0.5)
skew_ret_OS_SK=skew(port_return_OS_SK,0)
kurt_ret_OS_SK=kurtosis(port_return_OS_SK,0)


sd_data_SK = {"Annualized return":[exp_IS_SK,exp_OS_SK],"Volatility":[sd_re_IS_SK,sd_re_OS_SK],"Skewness":[skew_ret_IS_SK,skew_ret_OS_SK],"Kurtosis":[kurt_ret_IS_SK,kurt_ret_OS_SK]}
sd_OS = pd.DataFrame(sd_data_SK,index=['In-sample Portfolio','Out-sample Portfolio'])
sd_OS.to_latex("table/comparison_SK.tex")

### Other graphs ###

# cumulative returns of different assets (all dataset)
assets_cumul_return = (simpleReturns_OS+1).cumprod()
assets_cumul_return = assets_cumul_return*100
plt.rcParams["figure.figsize"] = (15,10)
assets_cumul_return.plot()
plt.legend(assets_cumul_return.columns,loc='lower left',fontsize='large')
plt.ylabel('Performance')
plt.xlabel('Date')
plt.savefig('fig/asset_cumul_return.png')


# cumulative returns of different assets ()
dfCumulRet = ((df.iloc[:,1:]/df.shift(1).iloc[:,1:] - 1).dropna() + 1).cumprod()
dfCumulRet = dfCumulRet*100
dfCumulRet.plot()
plt.legend(dfCumulRet.columns,loc='best',fontsize='large')
plt.ylabel('Performance')
plt.xlabel('Date')
plt.savefig('fig/asset_cumul_return_outSample.png')

assets_cumul_return_IS = (simpleReturns_IS+1).cumprod()
assets_cumul_return_IS = assets_cumul_return_IS*100
plt.rcParams["figure.figsize"] = (15,10)
assets_cumul_return_IS.plot()
plt.legend(assets_cumul_return.columns,loc='upper left',fontsize='large')
plt.ylabel('Performance')
plt.xlabel('Date')
plt.savefig('fig/asset_cumul_return_IN.png')