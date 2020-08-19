'''
Created on 2020/06/24

@author: dlzpj
'''
class calculateAQI(object):
    '''
    calculate AQI with Taiwan standard 
    
    
    '''


    def __init__(self,air_data):
        
        """
        air_data:{'Ox_8':.0,'Ox':.0,'PM10':.0,'PM25':.0,'CO':.0,'SO2':.0,'SO2_24':.0,'NO2':.0}
        """
        self.MAX_AQI=0
        self.air_data=air_data
        self.calculatetAQI()
            
 
    
    def calculatetAQI(self):
        '''
        Constructor
        '''
        self.AQI = []
        myAQI={'Ox_8':.0,'Ox':.0,'PM10':.0,'PM25':.0,'CO':.0,'SO2':.0,'SO2_24':.0,'NO2':.0}
        self.rank_rule=50
        
        for air in self.air_data.keys():
            myAQI[air]=self.air_data[air]
        
       
        AQI_Ox_8=0
        if myAQI['Ox_8']<=0.054:
            AQI_Ox_8 =50/(0.054)*(myAQI['Ox_8'])
        elif myAQI['Ox_8']<=0.070:
            AQI_Ox_8=49/(0.015)*(myAQI['Ox_8']-0.055)+51
        elif myAQI['Ox_8']<=0.085:
            AQI_Ox_8=49/(0.085-0.071)*(myAQI['Ox_8']-0.071)+101
        elif myAQI['Ox_8']<=0.105:
            AQI_Ox_8=49/(0.105-0.086)*(myAQI['Ox_8']-0.086)+151
        elif myAQI['Ox_8']<=0.200:
            AQI_Ox_8=49/(0.200-0.106)*(myAQI['Ox_8']-0.106)+201
        self.AQI.append(AQI_Ox_8)
        
        AQI_Ox=0
        if myAQI['Ox']<=0.125:
            AQI_Ox =0
        elif myAQI['Ox']<=0.164:
            AQI_Ox=49/(0.164-0.125)*(myAQI['Ox']-0.125)+101
        elif myAQI['Ox']<=0.204:
            AQI_Ox=49/(0.204-0.165)*(myAQI['Ox']-0.165)+151
        elif myAQI['Ox']<=0.404:
            AQI_Ox=49/(0.404-0.205)*(myAQI['Ox']-0.205)+201
        elif myAQI['Ox']<=0.504:
            AQI_Ox=49/(0.504-0.405)*(myAQI['Ox']-0.405)+301
        elif myAQI['Ox']<=0.504:
            AQI_Ox=49/(0.604-0.505)*(myAQI['Ox']-0.505)+401
        self.AQI.append(AQI_Ox)
        
        
        AQI_PM10=0
        if myAQI['PM10']<=54:
            AQI_PM10 =50/(54)*(myAQI['PM10'])
        elif myAQI['PM10']<=154:
            AQI_PM10=49/(99)*(myAQI['PM10']-55)+51
        elif myAQI['PM10']<=254:
            AQI_PM10=49/(99)*(myAQI['PM10']-155)+101
        elif myAQI['PM10']<=354:
            AQI_PM10=49/(99)*(myAQI['PM10']-255)+151
        elif myAQI['PM10']<=424:
            AQI_PM10=49/(69)*(myAQI['PM10']-355)+201
        elif myAQI['PM10']<=504:
            AQI_PM10=49/(89)*(myAQI['PM10']-425)+301
        elif myAQI['PM10']<=604:
            AQI_PM10=49/(99)*(myAQI['PM10']-505)+401
        self.AQI.append(AQI_PM10)
        
        
        AQI_PM25=0
       
        if myAQI['PM25']<=15.4:
            AQI_PM25 =50/(15.4)*(myAQI['PM25'])
        elif myAQI['PM25']<=35.4:
            AQI_PM25=49/(35.4-15.5)*(myAQI['PM25']-15.5)+51
        elif myAQI['PM25']<=54.4:
            AQI_PM25=49/(54.4-35.5)*(myAQI['PM25']-40.5)+101
        elif myAQI['PM25']<=150.4:
            AQI_PM25=49/(150.4-54.5)*(myAQI['PM25']-65.5)+151
        elif myAQI['PM25']<=250.4:
            AQI_PM25=49/(250.4-150.5)*(myAQI['PM25']-150.5)+201
        elif myAQI['PM25']<=350.4:
            AQI_PM25=49/(350.4-250.5)*(myAQI['PM25']-250.5)+301
        elif myAQI['PM25']<=500.4:
            AQI_PM25=49/(500.4-350.5)*(myAQI['PM25']-350.5)+401
        self.AQI.append(AQI_PM25)
        
        AQI_CO=0
        if myAQI['CO']<=4.4:
            AQI_CO =50/(4.4)*(myAQI['CO'])
        elif myAQI['CO']<=9.4:
            AQI_CO=49/(4.9)*(myAQI['CO']-4.5)+51
        elif myAQI['CO']<=12.4:
            AQI_CO=49/(2.9)*(myAQI['CO']-9.5)+101
        elif myAQI['CO']<=15.4:
            AQI_CO=49/(2.9)*(myAQI['CO']-12.5)+151
        elif myAQI['CO']<=30.4:
            AQI_CO=49/(14.9)*(myAQI['CO']-15.5)+201
        elif myAQI['CO']<=40.4:
            AQI_CO=49/(9.9)*(myAQI['CO']-30.5)+301
        elif myAQI['CO']<=50.4:
            AQI_CO=49/(9.9)*(myAQI['CO']-40.5)+401
        self.AQI.append(AQI_CO)
        
        
        AQI_SO2=0
        if myAQI['SO2']<=0.035:
            AQI_SO2 =50/(0.035)*(myAQI['SO2'])
        elif myAQI['SO2']<=0.075:
            AQI_SO2=49/(0.075-0.036)*(myAQI['SO2']-0.036)+51
        elif myAQI['SO2']<=0.185:
            AQI_SO2=49/(0.185-0.076)*(myAQI['SO2']-0.076)+101
       
        self.AQI.append(AQI_SO2)   
        
        
        AQI_SO2_24=0
        if myAQI['SO2_24']<0.186:
            AQI_SO2_24=0
        elif myAQI['SO2_24']<=0.304 and myAQI['SO2_24']>=0.186:
            AQI_SO2_24=49/(0.304-0.186)*(myAQI['SO2_24']-0.186)+151
        elif myAQI['SO2_24']<=0.604:
            AQI_SO2_24=49/(0.604-0.305)*(myAQI['SO2_24']-0.305)+201
        elif myAQI['SO2_24']<=0.804:
            AQI_SO2_24=49/(0.804-0.605)*(myAQI['SO2_24']-0.605)+301
        elif myAQI['SO2_24']<=1.004:
            AQI_SO2_24=49/(1.004-0.805)*(myAQI['SO2_24']-0.805)+401
            
        self.AQI.append(AQI_SO2_24)      
            
        AQI_NO2=0
        if myAQI['NO2']<=0.053:
            AQI_NO2 =50/(0.053)*(myAQI['NO2'])     
        elif myAQI['NO2']<=0.100:
            AQI_NO2=49/(0.100-0.054)*(myAQI['NO2']-0.054)+51
        elif myAQI['NO2']<=0.360:
            AQI_NO2=49/(0.360-0.101)*(myAQI['NO2']-0.360)+101
        elif myAQI['NO2']<=0.649:
            AQI_NO2=49/(0.649-0.361)*(myAQI['NO2']-0.361)+151
        elif myAQI['NO2']<=1.249:
            AQI_NO2=49/(1.249-0.650)*(myAQI['NO2']-0.650)+201
        elif myAQI['NO2']<=1.649:
            AQI_NO2=49/(1.649-1.250)*(myAQI['NO2']-1.250)+301
        elif myAQI['NO2']<=2.049:
            AQI_NO2=49/(2.049-1.650)*(myAQI['NO2']-1.650)+401
            
        self.AQI.append(AQI_NO2)
        self.MAX_AQI=max(self.AQI)
    
    def getAQI(self):    
        '''
        Return: float, the value of Taiwan AQI
        
        '''     
        return  self.MAX_AQI
    
    
    
    
    def getAQIRank(self):
        '''
        Return: int, the rank of Taiwan AQI
        
        '''
        
        Rank = 0
        if self.MAX_AQI/self.rank_rule<4.0:
            Rank = int(self.MAX_AQI/50)
        elif self.MAX_AQI/self.rank_rule<=6.0:
            Rank = int(4)
        else :
            Rank = int(5)
        
        return Rank