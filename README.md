# pytorch-AWAC

Advantage weighted Actor Critic for Offline RL implemented in pytorch.

A cleaner implementation for AWAC built on top of [spinning_up](https://github.com/openai/spinningup) SAC, and [rlkit](https://github.com/vitchyr/rlkit/tree/master/examples/awac).


If you use this code in your research project please cite us as:
```
@misc{Sikchi_pytorchAWAC,
author = {Sikchi, Harshit and Wilcox, Albert},
title = {{pytorch-AWAC}},
url = {https://github.com/hari-sikchi/AWAC}
}
```


## Running the code

```
python run_agent.py --env <env_name>  --seed <seed_no>  --exp_name <experiment name> --algorithm 'AWAC'
```


## Plotting
```
python plot.py <data_folder> --value <y_axis_coordinate> 
```

The plotting script will plot all the subfolders inside the given folder. The value is the y-axis that is required.
'value' can be:
* AverageTestEpRet


## Results
Environment: Ant-v2 
Dataset: https://drive.google.com/file/d/1edcuicVv2d-PqH1aZUVbO5CeRq3lqK89/view
Plot [Compare to Figure 5 of the official paper]: 
![image](https://user-images.githubusercontent.com/16147295/114596611-77e98500-9cad-11eb-8e6f-1c1b9a0b965f.png)


## References

@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}

@article{nair2020accelerating,
  title={Accelerating Online Reinforcement Learning with Offline Datasets},
  author={Nair, Ashvin and Dalal, Murtaza and Gupta, Abhishek and Levine, Sergey},
  journal={arXiv preprint arXiv:2006.09359},
  year={2020}
}
