export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt

pip install torch==1.13.0 torch-scatter==2.1.0 torchvision==0.14.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html 

python neugraspnet/scripts/convonet_setup.py build_ext --inplace

cd neugraspnet/neugraspnet
python -u scripts/test/sim_grasp_multiple.py --num-view 1 --object_set pile/test --scene pile --num-rounds 100 --model ./data/networks/neugraspnet_pile_efficient.pt --resolution=64 --type neu_grasp_pn_deeper_efficient --qual-th 0.5 --max_grasp_queries_at_once 40 --result-path ./data/results/neu_grasp_pile_efficient --sim-gui

python -u scripts/test/sim_grasp_multiple.py --num-view 1 --object_set egad --scene egad --num-rounds 100 --model ./data/networks/neugraspnet_pile_efficient.pt --resolution=64 --type neu_grasp_pn_deeper_efficient --qual-th 0.5 --max_grasp_queries_at_once 40 --result-path ./data/results/neu_grasp_egad_efficient --sim-gui