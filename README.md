export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt

pip install torch==1.13.0 torch-scatter==2.1.0 torchvision==0.14.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html 

python neugraspnet/scripts/convonet_setup.py build_ext --inplace

python -u scripts/test/sim_grasp_multiple.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --model ./data/networks/neugraspnet_pile_efficient.pt --resolution=64 --type neu_grasp_pn_deeper4 --qual-th 0.5 --sim-gui --result-path ./data/results/neu_grasp_pile