import prep_input


Xin = 'x_fmo_7_adolph_coeff' 
Yin = 'y_fmo_7_adolph_coeff'
dataPaths = ['/home/dell/arif/pypackage/sb/data/training_data/sym', 
             '/home/dell/arif/pypackage/sb/data/training_data/asym', 
             '/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/training_data/combined', 
             '/mnt/disk1/qd3set/7_sites_cho_etal/training_set/combined',
             '/mnt/disk1/qd3set/8_sites_jia_etal/training_set/combined',
             '/mnt/disk1/qd3set/8_sites_olbrich_etal/training_data']


#dataPath = '/mnt/disk1/qd3set/fmo_trimer/training_set/init_8'
# Set seeds for reproducibility
Xin = 'x_fmo_7_adolph_coeff' 
Yin = 'y_fmo_7_adolph_coeff'

#x = Xin
#y = Yin
systemType = 'FMO' 
n_states = 7
energyNorm = 1.0
DeltaNorm = 1.0
gammaNorm = 500
lambNorm = 520 
tempNorm = 510 
time_step = 0.005
#dataPath =  '/home/dell/arif/coefficients/fmo_model/fmo_7_adolph/init_1'
#prep_input.OSTL_coeff(x, y,
#        systemType,
#        n_states,
#        energyNorm,
#        DeltaNorm,
#        gammaNorm,
#        lambNorm,
#        tempNorm,
#        dataPath)

Xin = 'x_fmo_7_adolph_normal_pinn' 
Yin = 'y_fmo_7_adolph_normal_pinn'
x = Xin
y = Yin
dataPath = '/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/training_data/init_1'
#dataPath = '/mnt/disk1/qd3set/heom_8-site_Busch_and_Olbric/training_data'
prep_input.OSTL(x, y,
        systemType,
        n_states,
        energyNorm,
        DeltaNorm,
        gammaNorm,
        lambNorm,
        tempNorm,
        dataPath)

