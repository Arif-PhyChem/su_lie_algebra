import prep_input

Xin = 'x_punn-pinn'
Yin = 'y_punn-pinn'

x = Xin
y = Yin
systemType = 'SB'
n_states = 2
energyNorm = 1.0
DeltaNorm = 1.0
gammaNorm = 10
lambNorm = 1.0
tempNorm = 1.0
time_step = 0.05
dataPath =  '/home/dell/arif/pypackage/sb/data/training_data/combined'  
prep_input.OSTL(x, y,
        systemType,
        n_states,
        energyNorm,
        DeltaNorm,
        gammaNorm,
        lambNorm,
        tempNorm,
        dataPath)


