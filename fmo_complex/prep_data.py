import prep_input




Xin = 'x_fmo_7_1_su7-punn-pinn' 
Yin = 'y_fmo_7_1_su7-punn-pinn'

x = Xin
y = Yin
systemType = 'FMO' 
n_states = 7
energyNorm = 1.0
DeltaNorm = 1.0
gammaNorm = 500  # normalization constant for gamma
lambNorm = 520   # normalization constant for lambda
tempNorm = 510   # normalization constant for temperature
time_step = 0.005
dataPath =  '/home/dell/arif/coefficients/fmo_model/fmo_7_adolph/init_1'
prep_input.OSTL_coeff(x, y,
        systemType,
        n_states,
        energyNorm,
        DeltaNorm,
        gammaNorm,
        lambNorm,
        tempNorm,
        dataPath)

Xin = 'x_fmo_7_1_punn-pinn' 
Yin = 'y_fmo_7_1_punn-pinn'
x = Xin
y = Yin
dataPath = '/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/training_data/init_1'
prep_input.OSTL(x, y,
        systemType,
        n_states,
        energyNorm,
        DeltaNorm,
        gammaNorm,
        lambNorm,
        tempNorm,
        dataPath)

