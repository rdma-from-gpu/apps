[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13332379.svg)](https://doi.org/10.5281/zenodo.13332379)



This folder contains a set of application to demonstrate the functionalities of our prototype.

#Requirements

- A NVIDIA GPU. The newer the better, it has been tested on a Tesla T4
- A NVIDIA Mellanox NIC. This has been tested on CX5 and CX6 in their 100G versions, but should work on any NIC supported by the `mlx5` driver in rdma-core.
- `nvidia_peermem` module loaded
- For predictable metrics, fixed CPU governor and performance mode on the GPU. See `../helpers/setup.sh` for some hints.
- For maximum performances, a PCIe switch between the NIC and the GPU

