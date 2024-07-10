from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

# if LOCAL:
#     N_EXPS_IN_PARALLEL = 5
# else:
#     N_EXPS_IN_PARALLEL = 3

N_CORES = 5 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 10000
PARTITION = 'dgx' # 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-6DoF'

name = 'pn_deeper_with_occ_random_view_seen_pc'

launcher = Launcher(
    exp_name='eval_'+name,
    exp_file='sim_grasp_multiple', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    # project_name='project01907',  # for hrz cluster
    n_seeds=N_SEEDS,
    # n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

# Experiment configs (In this case, they are all argparse arguments for the main python file)
launcher.add_experiment(
    num_view=1,
    num_rounds=100,
    resolution=64,
    
    type='neu_grasp_pn_deeper',
    # With_occ model: /home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-03-01-42-37_dataset=data_pile_train_constructed_4M_HighRes_radomized_views_no_table,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_no_tab_deeper_DIMS_WITH_occ/neural_grasp_neu_grasp_pn_deeper_183258.pt
    # No occ model: /home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-01-08-11-39_dataset=data_pile_train_constructed_4M_HighRes_radomized_views,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_deeper_DIMS_CONT/best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9097.pt
    model='/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/runs_relevant/23-05-03-01-42-37_dataset=data_pile_train_constructed_4M_HighRes_radomized_views_no_table,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_no_tab_deeper_DIMS_WITH_occ/neural_grasp_neu_grasp_pn_deeper_183258.pt',
    qual_th=0.5,
    object_set='pile/test',
    scene='pile',
    randomize_view=True, # Don't pass if False
    seen_pc_only=True, # Don't pass if False
    result_path='/home/jauhri/IAS_WS/potato-net/GIGA-TSDF/GIGA-6DoF/data/results/'+name,
    )

launcher.run(LOCAL, TEST)