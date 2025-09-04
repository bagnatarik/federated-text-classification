from utils import make_directory

if __name__=='__main__':

    # Create directories if not exists
    make_directory('data')
    make_directory('preprocessing')
    make_directory('models')
    make_directory('strategy')
    make_directory('evaluation')
    make_directory('results')
    make_directory('results', 'plots')
    make_directory('results', 'fl_runs')
    make_directory('results\\fl_runs', 'S1_FedAvg_IID_5')
    make_directory('results\\fl_runs', 'S2_FedAvg_nonIID_20')
    make_directory('results\\fl_runs', 'S3_Clipping')
    make_directory('results\\fl_runs', 'S4_Clipping_LowNoise')
    make_directory('results\\fl_runs', 'S5_Clipping_HighNoise')

