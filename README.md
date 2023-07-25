# Energy Community Planning

This repository presents the implementation code of the open-access journal article [Can occupant behaviors affect urban energy planning? Distributed stochastic optimization for energy communities](https://doi.org/10.1016/j.apenergy.2023.121589).


## Citation
If you find this code useful and use it in your work, please reference our article:

[Leprince, J., Schledorn, A., Guericke, D., Dominkovic, D., Madsen, H., and Zeiler, W., 2023. Can occupant behaviors affect urban energy planning? Distributed stochastic optimization for energy communities](https://doi.org/10.1016/j.apenergy.2023.121589)

```
BibTex:
@article{LEPRINCE2023121589,
title = {Can occupant behaviors affect urban energy planning? Distributed stochastic optimization for energy communities},
journal = {Applied Energy},
volume = {348},
pages = {121589},
year = {2023},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2023.121589},
url = {https://www.sciencedirect.com/science/article/pii/S0306261923009534},
author = {Julien Leprince and Amos Schledorn and Daniela Guericke and Dominik Franjo Dominkovic and Henrik Madsen and Wim Zeiler},
keywords = {Energy communities, District energy management, Optimal energy planning, Stochastic optimization, Occupant behavior, Demand side management},
}

```

### Repository contributors

Dr. [Julien Leprince](https://github.com/JulienLeprince),
[Amos Schledorn](https://github.com/amosschle)


## Repository structure
```
energycommunityplanning
└─ data
|   ├─ in                               <- input - scenario data
|   └─ out                              <- output - energy planning design strategies
├─ src
|   ├─ 1_scenario_generation            <- scenario seasonal bootstrapping and clustering
|   ├─ 2_main_distributed_problem       <- distributed stochastic optimization problem
|   ├─ 2_poc_centralized_problem        <- centralized stochastic optimization problem serving convergence proof of concept
|   ├─ 3_sensitivityanalysis            <- local sensitivity analysis
|   ├─ 4_results_visualization          <- result visualization code source
|   ├─ RC_models                        <- lumped resistance capacity model implementation
|   └─ parameters                       <- optimization problem parameter file
└─ README.md                            <- README for developers using this code
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details