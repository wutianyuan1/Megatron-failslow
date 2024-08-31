# Usage: mpirun --allow-run-as-root --oversubscribe -np 8 bash ./test_mpi.sh
echo $OMPI_COMM_WORLD_RANK
CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK python entrypoint.py $OMPI_COMM_WORLD_RANK