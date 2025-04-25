import prep_input

Xin = 'x_sb_punn_pinn'
Yin = 'y_sb_punn_pinn'

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


