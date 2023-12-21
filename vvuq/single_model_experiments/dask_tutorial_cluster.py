"""Dask jobqueue 30min tutorial script to demonstrate parallelisation."""

# from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress
import time

# Init cluster (describes a single node)
# cluster = SLURMCluster(
#     cores=1,
#     processes=1,  # check docs
#     memory="4GB",
#     account="UKAEA-AP001-CPU",
#     walltime="00:10:00",
#     queue="cclake",
# )
# cluster.scale(4)  # 16 workers (i.e. nodes?)
# print(cluster.job_script())

# Create Dask client associated with cluster
# Connect scheduler to workers
# client = Client(cluster)

# Code from now on submitted to batch queue


def slow_increment(x):
    time.sleep(1)
    return x + 1


# This is important to avoid starting new processes on import
if __name__ == "__main__":
    client = Client()
    print(client)
    futures = client.map(slow_increment, range(100))
    progress(futures)
    print("Finished running futures")
