from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 1

# if LOCAL:
#     N_EXPS_IN_PARALLEL = 5
# else:
#     N_EXPS_IN_PARALLEL = 3

N_CORES = 96 #N_EXPS_IN_PARALLEL # HRZ nodes have 96 cores
# MEMORY_SINGLE_JOB = 1000
# MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
MEMORY_PER_CORE = 500
# PARTITION = 'amd2,amd'  # 'amd', 'rtx'
# GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'GIGA-6DoF'

launcher = Launcher(
    exp_name='generate_PARTIAL_GPG_NOISY_surface_clouds',
    exp_file='generate_data_grasp_surface_clouds', # local path without .py
    # exp_file='/work/home/sj93qicy/IAS_WS/potato-net/GIGA-6DoF/scripts/generate_data_gpg_parallel', # without .py
    project_name='project01907',  # for hrz cluster
    n_seeds=N_SEEDS,
    # n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=8,
    minutes=59,
    seconds=0,
    # partition=PARTITION,
    # gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

# Experiment configs (In this case, they are all argparse arguments for the main python file)
launcher.add_experiment(
    raw_root="/work/scratch/sj93qicy/potato-net/data/pile/data_pile_train_random_raw_2M_GPG_MIXED",
    # root="/work/scratch/sj93qicy/potato-net/data/pile/data_pile_train_random_raw_4M_GPG_60_PARTIAL",
    # use_previous_scenes=True,
    # previous_root="/work/scratch/sj93qicy/potato-net/data/pile/data_pile_train_random_raw_4M_GPG_60",
    # scene="pile",
    # object_set="pile/train",
    num_proc=N_CORES,
    # grasps_per_scene=60,
    # grasps_per_scene_gpg=60, # i.e. all grasps are gpg grasps
    # partial_pc=True,
    # save_scene=True,
    # random=True
    save_occ_values=True, # Don't pass if False
    add_noise=True # Don't pass if False
    )

launcher.run(LOCAL, TEST)