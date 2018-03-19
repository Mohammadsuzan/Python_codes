# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:15:20 2018

@author: Mohammadsuzan.Shaikh
"""

def ks_calc(pred,actual,deciles):
    test=pd.concat([pred,actual],axis=1)
    test.columns=['predicted','dv']
    test['quantiles']=pd.qcut(test['predicted'],deciles,labels=False)
    ks=pd.concat([test[test['dv']==0].groupby(['quantiles']).agg({'dv':np.size}).rename(columns={'dv':'Non_Default'}),
          test[test['dv']==1].groupby(['quantiles']).agg({'dv':np.size}).rename(columns={'dv':'Default'})],axis=1)

    ks.fillna(0,inplace=True)
    ks['Non_Default_cumsum']=np.cumsum(ks['Non_Default']/np.sum(ks['Non_Default']))
    ks['Default_cumsum']=np.cumsum(ks['Default']/np.sum(ks['Default']))
    ks['ks']=ks['Non_Default_cumsum']-ks['Default_cumsum']
    return max(ks['ks']),ks
	
def auc_gini(dv,pred):
    dv=np.array(dv)
    pred=np.array(pred)
    auc=met.roc_auc_score(dv,pred)
    gini=(2*auc)-1
    return auc,gini