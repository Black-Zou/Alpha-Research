# -*- coding: utf-8 -*-
"""
一种构造零投资组合的矩阵方法，运用多线程提升代码的运算速度
方法来源：Kewei Hou,Chen Xue,Lu Zhang, 2018
         Replicating Anomalies
         The Review of Financial Studies

作者：    Black-Zou 245888725@qq.com
         Wenyi-Wang wenyiwang96@gmail.com
"""



import pandas as pd
import numpy as np
import threading


#可以获得子线程函数结果的线程类
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        threading.Thread.join(self) # wait
        try:
            return self.result
        except Exception:
            return None
#最大线程数
thread_max_num = threading.Semaphore(8)


#因子类
"""
输入：
     df：一张包含股票id(permno)，日期(yyyymm,int格式)，排序变量的dataframe
     ret_mon：一张包含股票id(permno)，日期(yyyymm,int格式)，股票月度收益率(adj_cret)，
              交易所代码(exchcd)的dataframe
     f：输出文件根目录
"""
class abnormal(object):
    def __init__(self,df,ret_mon,f):
        self.df = df
        #因子月度均值表
        self.stats = pd.DataFrame(columns = ['NYSE_VW', 'NYSE_EW', 'ALL_VW', 'ALL_EW'])
        #收益率矩阵
        self.ret_tab = ret_mon.pivot(index='date',columns='permno',values='adj_cret').iloc[1:]
        #权重矩阵
        self.oweights = ret_mon.pivot(index='date',columns='permno',values='size').shift().iloc[1:]
        self.weights = self.oweights.fillna(0)
        #综合结果表
        self.result = pd.DataFrame()
        #进程锁
        self.lock = threading.Lock()
        self.tasks = []
        self.str_lst = ['','6','12']
        #输出目录
        self.path = f+'/factor'
    
    """
    主函数
    输入：
         factor_name：排序变量名(str)
         start_date：因子起始日期(yyyymm,int)
         nbf: breakpoint分类器，1表示NYSE，0表示ALL
         vw: 组合权重分类器，1表示VW，0表示EW
         flag:特殊因子分类器，默认按照3/10分位数与7/10分位数分组
         group：分几组，默认10组
         data_frequency: 数据频率，quarterly表示频率高于年度，annual表示年度数据
    输出：
         根据分类标准与构造权重生成的四张csv表格：
             NYSE_VW：选择NYSE的breakpoint构造value-weighted投资组合
             NYSE_EW:选择NYSE的breakpoint构造equal-weighted投资组合
             ALL_VW: 选择全市场的breakpoint构造value-weighted投资组合
             ALL_EW: 选择全市场的breakpoint构造equal-weighted投资组合

         
    """
    def mimic(self,factor_name,start_date,nbf,vw,flag='quantile',group=10,data_frequency='quarterly'):
        str1 = 'NYSE_'*nbf+'ALL_'*(1-nbf)+'VW_'*vw+'EW_'*(1-vw)+factor_name#列名
        #变量矩阵，滞后一期排序变量，每月初我们使用上月变量对股票进行排序
        df_tab = self.df.pivot(index='date', columns='permno', values=factor_name).shift().iloc[1:]
        if nbf:
            #使用NYSE的breakpoint
            df_tab_quantile = self.df[self.df.exchcd==1]\
                                  .pivot(index='date', columns='permno', values=factor_name)\
                                  .shift().iloc[1:]
        else:
            #ALL-breakpoint,使用原矩阵
            df_tab_quantile = df_tab.copy()
        if vw:
            #使用股票市值作为权重
            weights = self.weights*(~np.isnan(self.ret_tab))
        else:
            #等权重
            weights = (self.weights/self.weights).fillna(0)*(~np.isnan(self.ret_tab))
        #将收益率矩阵中的空值填为0
        ret_tab = self.ret_tab.fillna(0)
        if flag == 'quantile':
            #默认投资组合构造方法
            hbp = df_tab_quantile.quantile((group-1)/group,axis=1)#high
            lbp = df_tab_quantile.quantile(1/group,axis=1)#low
        elif flag == 'rank':
            #当排序变量是从1开始的排序时
            hbp = len(np.unique(df_tab_quantile))/group*(group-1)#high
            lbp = len(np.unique(df_tab_quantile))/group#low
        elif flag == 'Nsi':
            #当变量需要按照大于0小于0分开分组时
            hbp = pd.DataFrame(np.where(df_tab_quantile>0, df_tab_quantile, np.nan)).quantile(6/7, axis=1)
            lbp = pd.DataFrame(np.where(df_tab_quantile<0, df_tab_quantile, np.nan)).quantile(1/2, axis=1)
        elif flag == 'F':
            #当变量是F-score时
            hbp = pd.Series([8]*len(df_tab))
            lbp = pd.Series([2]*len(df_tab))
        elif flag == 'G':
            #当变量是G-score时
            hbp = pd.Series([7]*len(df_tab))
            lbp = pd.Series([1]*len(df_tab))
        h_m = df_tab.sub(hbp.values,axis=0)>=0#筛选long的投资组合
        l_m = df_tab.sub(lbp.values,axis=0)<=0#筛选short的投资组合
        #计算long的收益矩阵
        factor_h = np.array(np.matrix(h_m)*np.matrix((weights.T)*(ret_tab.T)))/np.array(np.matrix(h_m)*np.matrix(weights.T))
        #取对角线坐标
        diagon = np.diag_indices(len(factor_h))
        #计算short的收益矩阵
        factor_l = np.array(np.matrix(l_m)*np.matrix((weights.T)*(ret_tab.T)))/np.array(np.matrix(l_m)*np.matrix(weights.T))
        #计算因子收益率
        factor = (factor_h-factor_l)
        #计算持有期为6个月的因子收益率
        factor6 = np.array(pd.DataFrame(factor).rolling(window=6,min_periods=6).mean())[diagon]
        #计算持有期为12个月的因子收益率
        factor12 = np.array(pd.DataFrame(factor).rolling(window=12,min_periods=12).mean())[diagon]
        #取对角线，为1个月持有期的投资组合收益率
        factor = factor[diagon]
        #构造输出表格
        factor_df = pd.DataFrame(index = df_tab.index)
        factor_df[str1] = factor
        if data_frequency == 'annual':
            pass
        else:
            factor_df[str1+'6'] = factor6
            factor_df[str1+'12'] = factor12
            factor_df.loc[start_date:,str1+'6'].iloc[:5] = np.nan
            factor_df.loc[start_date:,str1+'12'].iloc[:11] = np.nan
        #输出
        factor_df.loc[start_date:].to_csv(self.path+'/ex_fin_'+'NYSE_'*nbf+'ALL_'*(1-nbf)+'VW_'*vw+'EW_'*(1-vw)+factor_name+'.csv')
        with self.lock:
            #计入内存
            self.result = pd.concat([self.result,factor_df],axis=1)
            #记录均值，与原论文对比
            mean_lst = ((factor_df.loc[start_date:201612].mean()*100).round(2)).tolist()
            for i in range(len(mean_lst)):
                self.stats.loc[factor_name+self.str_lst[i],'NYSE_'*nbf+'ALL_'*(1-nbf)+'VW'*vw+'EW'*(1-vw)] = mean_lst[i]
        return factor_df
    
    """
    四张表格同时计算，输入同上
    """
    def auto(self,factor_name,start_date,flag='quantile',group=10,data_frequency='quarterly'):
        tasks = []
        for i in range(2):
            for j in range(2):
                vw = i
                nbf = j
                tasks.append(MyThread(self.mimic,args=(factor_name,start_date,nbf,vw,flag,group,data_frequency)))
                tasks[-1].start()