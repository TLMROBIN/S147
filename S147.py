from astropy.table import Table,Column
import numpy as np
import matplotlib.pyplot as plt
from cp_cice import *
def cal_CICEs(filtereddata, allfiltereddata, band1, band2, band3):
    data=Table.read(filtereddata,'wb')#得到样本的有效温度
    teff_0=data['teff']/1000
    
    [p, _, _, _, _] = fit_c(allfiltereddata, band1, band2)#根据全部数据输出拟合公式
    CI_0=Column(data[band1]-data[band2],name='CI')#得到样本的色指数
    ce12 = CI_0 - np.polyval(p, teff_0)#计算样本色余
    ce12.name = band1 + '-' + band2#改列名
    
    [p, _, _, _, _] = fit_c(allfiltereddata, band1, band3)
    CI_0=Column(data[band1]-data[band3],name='CI')
    
    ce13 = CI_0 - np.polyval(p, teff_0)

    ce13.name = band1 + '-' +  band3
    
    [p, _, _, _, _] = fit_c(allfiltereddata, band2, band3)
    CI_0=Column(data[band2]-data[band3],name='CI')
    ce23 = CI_0 - np.polyval(p, teff_0)
    ce23.name = band2 + '-' +  band3
    
    ce = Table([ce12 ,ce13,ce23])
    ce = ce[np.where(np.logical_and(np.logical_and(ce12 > 0, ce13 > 0), ce23 > 0))]
    return ce
if __name__=='__main__':

    
     
    
    newdata.write('E:\\BNUCloud\\work\\Data\\{0}_U4L3_filtered.fits'.format(zonename),overwrite=True)
    filtereddata='E:\\BNUCloud\\work\\Data\\{0}_U4L3_filtered.fits'.format(zonename)
    allfiltereddata = 'E:\\BNUCloud\\work\\Data\\U4L3_filtered.fits'#全部数据所在位置
    
    band1='g'
    band2='r'
    band3='i'
    ce=cal_CICEs(filtereddata, allfiltereddata, band1, band2, band3)
    
    plotce(ce[band1 + '-' + band3], ce[band1 + '-' + band2],name=samplename)
    plotce(ce[band2 + '-' + band3], ce[band1 + '-' + band2],name=samplename)
    plotce(ce[band1 + '-' + band3], ce[band2 + '-' + band3],name=samplename)
    plt.show()

    
        #else:
        #fitsdata='E:\\BNUCloud\\work\\Data\\S147_U4L3.fits'#外部获取
        #newdata,num=datafilter(fitsdata , magerror = Merror*100)#这里误差的单位是centimag所以数上限字扩大为100倍 