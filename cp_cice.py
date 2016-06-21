from astropy.table import Table,Column,vstack
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def datafilter(fitsdata,lowlimit=None,uplimit=None,magerror=None):
    "默认为 研究波段band= ('B', 'V', 'g', 'r', 'i'),\n\
    星等误差banderr=('Be', 'Ve', 'ge', 're', 'ie'),\n\
    星等误差限banderrc=0.05,\n\
    有效温度范围teffrange=(3800, 10000),\n\
    有限温度误差限tefferr=300,\n\
    logg范围loggrange=(4,1000),\n\
    logg误差限loggerr=0.5,\n\
    金属性范围fehrange=(-0.5, 0.5)\n\
    适用于UCAC4光学BVgri波段与LAMOST交叉结果，其他波段需自定义波段及误差\n\
    若测光数据并非来自LAMOST DR3,有效温度等列标题名称需酌情更改"
    #数据的上下限范围写入字典
    if magerror==None:
        magerror = 0.05
    if lowlimit==None:
        lowlimit=dict((('B',0),('V',0),('r',0),('i',0),('g',0),('Be',0),( 'Ve',0),( 'ge',0) ,('re',0),( 'ie',0),('teff',3800),('teff_err',0),('logg',4),('logg_err',0),('feh',-0.5)))
    if uplimit==None:
        uplimit=dict((('B',100),('V',100),('r',100),('i',100),('g',100),('Be',magerror),( 'Ve',magerror),( 'ge',magerror) ,('re',magerror),( 'ie',magerror),('teff',8400),('teff_err',300),('logg',20),('logg_err',0.5),('feh',0.5)))
    dtable = Table.read(fitsdata, 'wb')#读取数据表格
    print("colnames,if they didn't match your data, please creat a new dictionary\n",dtable.colnames, '\n')
    for item in lowlimit.keys():
        #选择数据满足条件的表格部份
        dtable=dtable[np.where(np.logical_and(dtable[item]>lowlimit[item],dtable[item]<uplimit[item]))]
        n = len(dtable)#行数
    return dtable, n
#newdata,num=datafilter(fitsdata)
#print('There are',num,'groups of data qualified')
#newdata.write('E:\\BNUCloud\\work\\Data\\U4L3_filtered.fits',overwrite=True)

def sigma3(filtereddata, band1, band2, teff=None, tbin=None, pnum=None, percentage=None):
    'The usage of this function is to del the data out of 3 sigma criteria from samples\n\
    specific CI will also be added to the data table\n\
    pecentage is the proportion of chosen points'
    if tbin==None:
        tbin=50
    if teff==None:
        teff='teff'
    if pnum==None:
        pnum=2
    if percentage==None:
        percentage=0.05
    data=Table.read(filtereddata,'wb')#读取数据表格
    #加入色指数列
    CIname='CI_{0}{1}'.format(band1, band2)
    color=Column(data[band1]-data[band2],name=CIname)
    data.add_column(color)#将原色指数列加入表data
    #重复的读写需要不能对原始数据进行多次相同修改
    teff_bin=np.ceil(data[teff]/tbin)#制造分组依据：每50K范围内的数据的teff_bin值相同
    data_grouped=data.group_by(teff_bin)#分bin
    n=len(data_grouped.groups)#bin的数量
    i=0
    for group in data_grouped.groups:#对每个组的数据进行处理 
        while True:
            group.sort([CIname])#对色指数列进行排序
            len1=len(group)#样本原长度
            top=np.floor(len1*percentage)#样本索引上限值
            snum=top+1#样本数量，因为python是从0开始索引，所以实际样本数比索引上限大1
            samples=group[0:snum]
            others=group[snum:len1]
            CI_0_mean=np.mean(samples[CIname])#前百分之五的均值
            CI_0_std=np.std(samples[CIname])#求出bin中下方百分之五样本的标准差
            cha=samples[CIname]-CI_0_mean
            samples=samples[np.where(abs(cha)<=3*CI_0_std)] #减去小于中值-3倍标准差的值，即下方异常的点
            group=vstack([others,samples])
            len2=len(group)
            if len1==len2:
                if i==0:
                    newgroup=group
                    i=1
                else:
                    newgroup=vstack([newgroup,group])
                break
    newgroup.write(filtereddata,overwrite=True) 

def fit_c(filtereddata, band1, band2, sigma=None, teff=None, tbin=None, pnum=None, percentage=None, expfit=None):
    'The usage of this function is to find the relation between teff and color with given teff bins.\n\
    pnum means the level for fitting of a polynomial.\n\
    pecentage is the proportion of chosen points \n\
    add 3sigma criteria and exponential fit using arguments to active them'
    if tbin==None:
        tbin=50
    if teff==None:
        teff='teff'
    if pnum==None:
        pnum=2
    if percentage==None:
        percentage=0.05
    if sigma==None:
        sigma='off'
    if expfit==None:
        expfit='off'
    #执行3sigma判据，添加在这里是为了使fit_c函数自身也能执行3sigma判断
    if sigma=='on':
        sigma3(filtereddata, band1, band2)    
    data=Table.read(filtereddata,'wb')#读取数据表格
    #若未使用3sigma判据，则需加入色指数列；若已使用则无需再加，否则会报重复
    #重复的读写需要不能对原始数据进行多次重复操作
    if sigma!='onbefore' and sigma!= 'on':#在本次运行函或此前已使用了3sigma判据，则无需再添加色指数列
        CIname='CI_{0}{1}'.format(band1, band2)
        color=Column(data[band1]-data[band2],name=CIname)
        data.add_column(color)#将色指数列加入表data
    CIname='CI_{0}{1}'.format(band1, band2)#3sigma判断函数和拟合函数共用的色指数列名
    data[teff]=data[teff]/1000
    teff_bin=np.ceil(data[teff]/(tbin/1000))#制造分组依据：每50K范围内的数据的teff_bin值相同
    data_grouped=data.group_by(teff_bin)#分bin
    n=len(data_grouped.groups)#bin的数量
    CISample='CI0_{0}{1}'.format(band1, band2)
    CI_0= Column(name=CISample, length=n)#制造一个新的列用于储存选取的色指数值
    i=0
    for group in data_grouped.groups:#对每个组的数据进行处理 
        group.sort([CIname])#对色指数列进行排序
        len1=len(group)
        top=np.floor(len1*percentage)#样本索引上限值
        snum=top+1#样本数量，因为python是从0开始索引，所以实际样本数比索引上限大1
        #print('第',i+1,'个bin中有',snum,'个样本')
        samples=group[0:snum]
        CI_0_mid=np.median(samples[CIname])#前百分之五的中值
        CI_0[i]=CI_0_mid#选取前百分之五的中值
        i=i+1
    teff0=data_grouped[teff]
    CI0=data_grouped[CIname]
    dat=data_grouped.groups.aggregate(np.median)#对各bin中的数据取中值，主要是对teff
    dat.add_column(CI_0)
    #筛选或未删选的数据存入dat，CI0、teff0为整体数据、dat[teff] , dat[CISample] 为样本数据
    #选择拟合方式
    #指数拟合
    if expfit=='on':
        def  expfunc(x, a, b, c):#构造拟合曲线
            return  a * np.exp(- x/b) + c
        p, _ = curve_fit(expfunc, dat[teff], dat[CISample])
        fomular="({0}-{1})0={2:.4f}*exp(-T/{3:.4f})+{4:.4f}".format(band1,band2,p[0],p[1],p[2])
        def efitval(p,t):
            return p[0]*np.exp(-t/p[1])+p[2]
        CI_f=efitval(p,dat[teff])#用样本的有效温度拟合色指数
        CIR=CI_f-dat[CISample]#得到拟合的残差
    #多项式拟合
    else:
        p=np.polyfit(dat[teff],dat[CISample],pnum)
        power=np.arange(len(p))[::-1]#逆序，用于公式中幂指数
        fomular="$({0}-{1})_0$=".format(band1,band2)
        i=0
        while i<len(p):
            #以下三个if皆是为了使得输出的公式形式符合习惯
            if i==len(p)-1:
                fomular=fomular+'{0:+.4f}'.format(p[i])
                i=i+1
            elif i==len(p)-2:
                fomular=fomular+'{0:+.4f}*$T_e$'.format(p[i])
                i=i+1
            elif i==0:
                fomular=fomular+'{0:.4f}*$T_e$^{1}'.format(p[i],power[i])
                i=i+1
            else:
                fomular=fomular+'{0:+.4f}*$T_e$^{1}'.format(p[i],power[i])
                i=i+1
        CI_f=np.polyval(p,dat[teff])#用样本的有效温度拟合色指数
        CIR=CI_f-dat[CISample]#得到拟合的残差
    return   p, dat[teff] , dat[CISample] , teff0 , CI0 , fomular , CI_f,  CIR

#filtereddata='E:\\BNUCloud\\work\\Data\\U4L3_filtered.fits'
#p,teff_s,CI_s,teff_0,CI_0=fit_c(filtereddata,'B','V')#p参数s样本0原始数据

def plot_tc(band1,band2,p,teff_s,CI_s,teff_0,CI_0, CI_f, CIR, fomular,name = 'Allsky'):
    'plot fit result and  residual error'
    plt.figure(figsize=(12,8))#打开画布
    plt.plot(teff_0,CI_0,'g.',markersize =1,label='Observed {0}-{1}'.format(band1,band2))#画出原始数据
    plt.plot(teff_s,CI_s,'b*',label='{0}-{1} samples'.format(band1,band2))#画出样本数据
    plt.plot(teff_s,CI_f,'r',label='Intrinsic {0}-{1}'.format(band1,band2),linewidth=2)#画出拟合曲线
    plt.xlabel('$T_e$/1000K',fontsize=30)
    plt.ylabel('Color Index/mag',fontsize=30)
    plt.title('{0} Intrinsic CI Fitting of {1}-{2}'.format(name, band1, band2),fontsize=32)
    plt.figtext(0.15,0.15,fomular,fontsize=19)
    plt.legend()
    filename='E:\\BNUCloud\\work\\S147_relatedpic\\{0} intrinsic {1}-{2}.png'.format(name,band1,band2)
    plt.savefig(filename)#保存图片
    plt.close()
    plt.figure(figsize=(12,8))#打开画布
    plt.bar(left = teff_s,height = CIR,width=0.05)
    plt.xlabel('$T_e$/1000K',fontsize=30)
    plt.ylabel('Residual errors/mag',fontsize=30)
    plt.title('{0} Residuals for Intrinsic CI Fit of {1}-{2}'.format(name, band1, band2),fontsize=32)
    filename='E:\\BNUCloud\\work\\S147_relatedpic\\{0}  residual errors for {1}-{2}.png'.format(name,band1,band2)
    plt.savefig(filename)#保存图片
    plt.close()

def cal_CE(band1, band2, p, teff_0, CI_0,expfit=None):#在直接给温度和色指数的情况下使用
    'calculate the ce  with given teff and CI'
    if expfit==None:
        expfit='off'
    
    if expfit=='on':#判断是否是指数拟合
        def efitval(p,t):
            return p[0]*np.exp(-t/p[1])+p[2]
        ce=CI_0 - efitval(p,teff_0)#计算色余 
    else: 
        ce = CI_0 - np.polyval(p, teff_0)#计算色余 
    ce.name = band1 + '-' + band2#改列名
    return ce

def cal_CE2(data, band1, band2, p, teffname=None):#在直接给整个数据的情况下使用
        'calculate CE for with given entire data'
        if teffname==None:
            teffname='teff'
        CI=Column(data[band1]-data[band2])
        teff=data[teffname]/1000
        ce=cal_CE(band1, band2, p, teff, CI)
        return ce  
    
def cipltce(filtereddata, band1, band2):
    'do all the works such as calculate the intrinsic CI foumular\n\
    plot the fit result, calculate the CE and give parameters for this CI'
    [p, teff_s, CI_s, teff_0, CI_0, fomular, CI_f,CIR] = fit_c(filtereddata, band1, band2, sigma='onbefore', pnum=3)#获得拟合参数p、样本数据s、整体数据0、公式、样本拟合结果f、残差R
    plot_tc(band1, band2, p, teff_s, CI_s, teff_0, CI_0, CI_f, CIR,fomular)#画出拟合结果和残差
    ce=cal_CE(band1, band2,p, teff_0, CI_0)
    return ce,p

def plotce(x,y, name = 'Allsky'):
    plt.figure(figsize=(12,8))
    p=np.polyfit(x,y,1)#直接拟合
    M = np.column_stack((x,)) # construct design matrix，用最小二乘法获得过零点拟合
    k, _, _, _ = np.linalg.lstsq(M, y) # least-square fit of M * k = y
    k=float(k)
    plt.plot(x,y,'b.',label='Data')
    plt.plot(x,x*k,'g',label='Linear Fitting through 0')
    plt.plot(x,np.polyval(p,x),'r',label='Linear Fitting')
    plt.figtext(0.15,0.85,'0 fit CE = {0:.4f}'.format(k), fontsize=22)
    plt.figtext(0.15,0.8,'directly fit CE = {0:.4f}'.format(p[0]), fontsize=22)
    plt.xlabel('E('+x.name+')',fontsize=30)
    plt.ylabel('E('+y.name+')',fontsize=30)
    plt.title(' {0} ({1})/({2})'.format(name, y.name,x.name),fontsize=32)
    filename='E:\\BNUCloud\\work\\S147_relatedpic\\{0} CE ({1}) vs ({2}).png'.format(name, x.name, y.name)
    plt.legend()
    plt.savefig(filename)
    print('when directly fit, {0}/{1}={2:.4f}'.format(y.name,x.name,p[0]))
    print('when fit through 0, {0}/{1}={2:.4f}\n'.format(y.name,x.name,k))
    
def delce(CE):
    for ce in CE.columns:
        CE = CE[np.where(CE[ce] > 0)]
    return CE
    
def zonefilter(data, zone, ra=None, dec=None, raname=None, decname=None ):
    'select sources in a zone'
    if ra==None:
        ra=84.75
    if dec==None:
        dec=27.8333
    if raname==None:
        raname='ra_1'
    if decname==None:
        decname='dec_1'
    #筛选距离目标中心一定范围内的源        
    newdata = data[np.where( (data[raname] -ra) **2 + (data[decname] - dec) **2 > zone[0] **2 )]
    newdata = data[np.where( (data[raname] -ra) **2 + (data[decname] - dec) **2 < zone[1] **2 )]
    num=len(newdata)
    print('There are',num,'groups of data qualified in this {0} to {1} degree zoon'.format(zone[0],zone[1]), '\n')
    return newdata

def twopltce(CE, band1, band2, textlocation, zero, color):
    'fit the CER in two ways but doesn\'t open a new figure . So a figure should be opened before'
    if zero==True:    
        #获取待求关系之色余及数据名称
        x, y = CE[band1],CE[band2]
        name = CE.meta['name']
        #用最小二乘法获得过零点拟合
        M = np.column_stack((x,)) # construct design matrix，
        k, _, _, _ = np.linalg.lstsq(M, y) # least-square fit of M * k = y
        k=float(k)
        plt.plot(x,x*k, color, label=name+' CER Linear Fitting through 0')
        plt.figtext(textlocation[0], textlocation[1],  '{0} CER = {1:.4f}'.format(name,k), fontsize=22)
        print('{0} CER fiting through 0 , {1}/{2}={3:.4f}'.format(name, y.name, x.name, k))
    else:
        #获取待求关系之色余及数据名称
        x, y = CE[band1],CE[band2]
        name = CE.meta['name']
        p=np.polyfit(x,y,1)#直接拟合
        plt.plot(x,np.polyval(p,x), color, label=name + ' CER Directly Linear Fitting')
        plt.figtext(textlocation[0], textlocation[1],  '{0} CER = {1:.4f}'.format(name, p[0]), fontsize=22)    
        print('{0} CER directly fiting , {1}/{2}={3:.4f}'.format(name, y.name, x.name, p[0]))
def plotce_3(CE1, CE2, CE3, band1, band2, zero):
    'plot 3 CER in one'
    plt.figure(figsize=(12,8))
    twopltce(CE1, band1, band2, [0.15, 0.85], zero, 'g')
    ##天区2
    ##
    twopltce(CE2, band1, band2, [0.15, 0.80], zero, 'r')
    ##天区3
    ##
    twopltce(CE3, band1, band2, [0.15, 0.75], zero, 'b')
    #绘图属性设置
    x, y = CE[band1],CE[band2]
    name1, name2, name3 = CE1.meta['name'], CE2.meta['name'], CE3.meta['name']
    plt.xlabel('E('+x.name+')',fontsize=30)
    plt.ylabel('E('+y.name+')',fontsize=30)
    plt.title('CER ({0})/({1}) Fitting'.format(y.name,x.name),fontsize=32)
    filename='E:\\BNUCloud\\work\\S147_relatedpic\\CER ({0})vs({1}) for 3 zone.png'.format(x.name, y.name)
    plt.legend()
    plt.savefig(filename)
    
        
if __name__ == '__main__':
    ##全天部分
    ##
    dataname='A9L3'
    fitsdata='E:\\BNUCloud\\work\\Data\\{0}.fits'.format(dataname)
    newdata,num=datafilter(fitsdata)
    print('There are',num,'groups of data qualified')
    newdata.write('E:\\BNUCloud\\work\\Data\\{0}_filtered.fits'.format(dataname),overwrite=True)
    filtereddata='E:\\BNUCloud\\work\\Data\\{0}_filtered.fits'.format(dataname)
    #启用五次3sigma拟合剔除所有不规范数据
    sigma3(filtereddata, 'B', 'V')
    sigma3(filtereddata, 'B', 'i')
    sigma3(filtereddata, 'V', 'i')
    sigma3(filtereddata, 'g', 'i')
    sigma3(filtereddata, 'r', 'i')
    #计算两个波段的内秉色指数拟合公式并画出拟合结果和残差图，以及计算相应色余
    ceBV, pBV = cipltce(filtereddata, 'B', 'V')
    ceBi, pBi = cipltce(filtereddata, 'B', 'i')
    ceVi, pVi = cipltce(filtereddata, 'V', 'i')
    cegi, pgi = cipltce(filtereddata, 'g', 'i')
    ceri, pri = cipltce(filtereddata, 'r', 'i')
    #整合色余并去除小于0的色余
    CE = Table([ceBV , ceBi , ceVi , cegi , ceri],meta={'name': ' Allsky'})
    CE = delce(CE)
    #画出色余比
    ColorExcess1 = 'B-V'
    ColorExcess2 = 'B-i'
    ColorExcess3 = 'V-i'
    ColorExcess4 = 'g-i'
    ColorExcess5 = 'r-i'
    plotce(CE[ColorExcess1], CE[ColorExcess2])
    plotce(CE[ColorExcess1], CE[ColorExcess3])
    plotce(CE[ColorExcess1], CE[ColorExcess4])
    plotce(CE[ColorExcess1], CE[ColorExcess5])
    ##分天区
    ##
    #S147
    #S147的半径    
    radius=[0,1.5] 
    Merror=0.1#星等误差限
    #总体筛选
    fitsdata='E:\\BNUCloud\\work\\Data\\U4L3.fits'
    newdata,num=datafilter(fitsdata , magerror = Merror)
    #筛选S147范围内的源
    S147=zonefilter(newdata,radius)
    #计算S147源的色余
    S147ceBV = cal_CE2(S147, 'B', 'V', pBV)
    S147ceBi = cal_CE2(S147, 'B', 'i', pBi)
    S147ceVi = cal_CE2(S147, 'V', 'i', pVi)
    S147cegi = cal_CE2(S147, 'g', 'i', pgi)
    S147ceri = cal_CE2(S147, 'r', 'i', pri)
    #整合色余并去0
    S147CE = Table([S147ceBV , S147ceBi , S147ceVi , S147cegi , S147ceri], meta={'name': ' S147'})
    S147CE = delce(S147CE)
    #画出色余比
    plotce(S147CE[ColorExcess1], S147CE[ColorExcess2], name = 'S147')
    plotce(S147CE[ColorExcess1], S147CE[ColorExcess3], name = 'S147')
    plotce(S147CE[ColorExcess1], S147CE[ColorExcess4], name = 'S147')
    plotce(S147CE[ColorExcess1], S147CE[ColorExcess5], name = 'S147')
    #S147周边环境
    #周边区间    
    interval=[2,5] 
    #筛选S147范围内的源
    Around=zonefilter(newdata,interval)
    #计算S147源的色余
    AroundceBV = cal_CE2(Around, 'B', 'V', pBV)
    AroundceBi = cal_CE2(Around, 'B', 'i', pBi)
    AroundceVi = cal_CE2(Around, 'V', 'i', pVi)
    Aroundcegi = cal_CE2(Around, 'g', 'i', pgi)
    Aroundceri = cal_CE2(Around, 'r', 'i', pri)
    #整合色余并去0
    AroundCE = Table([AroundceBV , AroundceBi , AroundceVi , Aroundcegi , Aroundceri], meta={'name': ' Around'})
    AroundCE = delce(AroundCE)
    #画出色余比
    plotce(AroundCE[ColorExcess1], AroundCE[ColorExcess2], name = 'Around')
    plotce(AroundCE[ColorExcess1], AroundCE[ColorExcess3], name = 'Around')
    plotce(AroundCE[ColorExcess1], AroundCE[ColorExcess4], name = 'Around')
    plotce(AroundCE[ColorExcess1], AroundCE[ColorExcess5], name = 'Around')

    plotce_3(CE, S147CE, AroundCE, ColorExcess1, ColorExcess2, True)
    plotce_3(CE, S147CE, AroundCE, ColorExcess1, ColorExcess3, True)
    plotce_3(CE, S147CE, AroundCE, ColorExcess1, ColorExcess4, True)
    plotce_3(CE, S147CE, AroundCE, ColorExcess1, ColorExcess5, True)
    
    
    
    plt.close('all')