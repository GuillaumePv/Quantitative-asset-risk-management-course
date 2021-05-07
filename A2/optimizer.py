import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize

def SK_criterion(weight,Lambda_RA,Returns_data):
    """
    
    this function computes the expected utility in the Markowitz case when 
    investors have a preference for skewness and kurtosis 
    Parameters
    ----------
    weight : list of float
        weights in the investor's portfolio.
    Lambda_RA : int
        the risk aversion parameter.
    Returns_data : list of float
        the set of returns.

    Returns
    -------
    criterion : list of float
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
    
    this function computes the expected utility in the Markowitz case when 
    investors have a preference for mean and variance
    Parameters
    ----------
   weight : list of float
        weights in the investor's portfolio.
    Lambda_RA : int
        the risk aversion parameter.
    Returns_data : list of float
        the set of returns.

    Returns
    -------
    criterion : list of float 
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

def Optimizer(returnData,num_col, opt='EV'):
    assert opt != "EV" or opt != 'SK'
    """
    

    Parameters
    ----------
    returnData : list of float
        the set of return datas.
    
    opt: which optimizer (EV or SK)

    Returns
    -------
    resultat : Object
        result of chosen optimizer
    """
    #starting points => 1% in each stock
    # weight for criterion :)
    # à modifier pour être utiliser selon le nombre d'asset
    x0 = np.array([x for x in range(num_col)])+0.01
    
    # constraint for weight
    cons=({'type':'eq', 'fun': lambda x:sum(x)-1},
    {'type':'ineq', 'fun': lambda x: x[1]},
    {'type':'ineq', 'fun': lambda x: x[2]- 0.01})
    Bounds= [(0 , 1) for i in range(0,num_col)] # boudns of weights -> 0 to 1
    
    Lambda_RA=3 #define teh risk aversion parameter
    
    if opt=='EV':
        resultat = minimize(EV_criterion, x0, method='SLSQP', args=(Lambda_RA,np.array(returnData.iloc[:,0:])),bounds=Bounds,constraints=cons,options={'disp': True})
    #res_SK.x: give the optimal weight
    else:
        resultat = minimize(SK_criterion, x0, method='SLSQP', args=(Lambda_RA,np.array(returnData.iloc[:,0:])),bounds=Bounds,constraints=cons,options={'disp': True})
    
    return resultat