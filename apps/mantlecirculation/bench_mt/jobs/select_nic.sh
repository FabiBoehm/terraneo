#!/bin/bash
# Pin each local rank to its GPU's PXB-adjacent IB HCA (see nvidia-smi topo -m):
#   GPU0->mlx5_3  GPU1->mlx5_2  GPU2->mlx5_5  GPU3->mlx5_4   (local rank == device id)
case ${OMPI_COMM_WORLD_LOCAL_RANK:-0} in
  0) export UCX_NET_DEVICES=mlx5_3:1 ;;
  1) export UCX_NET_DEVICES=mlx5_2:1 ;;
  2) export UCX_NET_DEVICES=mlx5_5:1 ;;
  3) export UCX_NET_DEVICES=mlx5_4:1 ;;
esac
export UCX_IB_GPU_DIRECT_RDMA=yes
exec "$@"
