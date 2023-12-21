# %% [markdown]
# # Single model (`palph2()`) UQ using Dask
#
# Try to capture the input and output uncertainties for a single model in Process. This will be done by sampling the input epistemic uncertainty parameter space to the code as a whole and capturing the input distributions to the model of interest. These input distributions can then be propagated through the model of interest, and the output variance analysed. This allows the uncertainty contribution of an individual model to be assessed, given a set of epistemic uncertainty inputs for the code as a whole.
#
# Dask will be used to parallelise the evaluations.

# %%
import easyvvuq as uq
import chaospy as cp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


# %% [markdown]
# ## Campaign to capture input distribution for `palph2()`
#
# Using the epistemic uncertain inputs for the entire code, capture the distribution of inputs for the `palph2()` subroutine.
#
# To start with, make just 2 inputs uncertain (for running locally).

# %%
# Init cluster (describes a single node, or less if need less than that per worker)
# cluster = SLURMCluster(
#     cores=56,
#     processes=4,  # check docs
#     memory="192GB",
#     account="UKAEA-AP001-CPU",
#     walltime="01:00:00",
#     queue="cclake",
# )

# Need less than a full node per worker
# Also, activate Singularity env and conda env on each worker node
cluster = SLURMCluster(
    cores=1,
    processes=1,
    memory="4GB",
    account="UKAEA-AP001-CPU",
    walltime="00:20:00",
    queue="cclake",
    python="singularity exec ~/process.sif bash ~/activate_py.sh",
)
cluster.scale(4)  # 16 workers
# print(cluster.job_script())

# Connect Dask client to remote cluster
client = Client(cluster)
# Code from now on submitted to batch queue

# %% [markdown]
# Execute the same as before, just with a `client` argument in the actions.

# %%
# Create campaigns dir if required
WORK_DIR_NAME = "campaigns"
Path(WORK_DIR_NAME).mkdir(exist_ok=True)

# Define campaign
campaign = uq.Campaign(name="model_inputs", work_dir=WORK_DIR_NAME)

# Define parameter space
# Uncertainties from Alex's SA paper

params = {
    "fgwped": {
        "type": "float",
        "min": 1.1,
        "max": 1.3,
        "default": 1.1,
    },  # check: not sure if this is right var
    "hfact": {"type": "float", "min": 1.0, "max": 1.2, "default": 1.1},
    "coreradius": {"type": "float", "min": 0.45, "max": 0.75, "default": 0.75},
    "fimp_2": {"type": "float", "min": 0.085, "max": 0.115, "default": 0.1},  # check
    "fimp_14": {
        "type": "float",
        "min": 1.0e-5,
        "max": 1.0e-4,
        "default": 5e-5,
    },  # check
    "psepbqarmax": {"type": "float", "min": 8.7, "max": 9.7, "default": 9.2},
    "flhthresh": {"type": "float", "min": 0.85, "max": 1.15, "default": 1.15},  # check
    "bscfmax": {"type": "float", "min": 0.95, "max": 1.05, "default": 0.99},
    "peakfactrad": {"type": "float", "min": 2.0, "max": 3.5, "default": 3.33},
    "kappa": {"type": "float", "min": 1.8, "max": 1.9, "default": 1.8},  # check default
    "etaech": {"type": "float", "min": 0.3, "max": 0.5, "default": 0.4},
    "feffcd": {"type": "float", "min": 0.5, "max": 5.0, "default": 1.0},
    "etath": {"type": "float", "min": 0.36, "max": 0.4, "default": 0.375},
    "etaiso": {"type": "float", "min": 0.75, "max": 0.95, "default": 0.9},
    "boundl_18": {
        "type": "float",
        "min": 3.25,
        "max": 3.75,
        "default": 3.5,
    },  # q^95_min
    "pinjalw": {"type": "float", "min": 51.0, "max": 61.0, "default": 51.0},
    "alstroh": {"type": "float", "min": 6.0e8, "max": 7.2e8, "default": 6.6e8},
    "sig_tf_wp_max": {
        "type": "float",
        "min": 5.2e8,
        "max": 6.4e8,
        "default": 5.8e8,
    },  # winding pack, or casing?
    "aspect": {"type": "float", "min": 3.0, "max": 3.2, "default": 3.1},
    "boundu_2": {
        "type": "float",
        "min": 11.0,
        "max": 12.0,
        "default": 11.5,
    },  # B_T^max: check default
    "triang": {"type": "float", "min": 0.4, "max": 0.6, "default": 0.5},
    "out_file": {"type": "string", "default": "out.csv"},
}

# QoIs for palph2(): inputs (args)
palph2_inputs = [
    "bt",
    "bp",
    "dene",
    "deni",
    "dnitot",
    "falpe",
    "falpi",
    "palpnb",
    "ifalphap",
    "pchargepv",
    "pneutpv",
    "ten",
    "tin",
    "vol",
    "palppv",
]

# Create encoder and decoder
encoder = uq.encoders.GenericEncoder(
    template_fname="baseline_2018.template", target_filename="IN.DAT"
)
decoder = uq.decoders.JSONDecoder(
    target_filename="before.json", output_columns=palph2_inputs
)

cmd = "process -i IN.DAT"
actions = uq.actions.local_execute(encoder, cmd, decoder)

# Add the app
campaign.add_app(name="single_model", params=params, actions=actions)

# Create PCE sampler
vary = {
    "aspect": cp.Uniform(3.0, 3.2),
    # "boundu_2": cp.Uniform(11.0, 12.0),
    # "psepbqarmax": cp.Uniform(8.7, 9.7),
}
pce_sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)

# Add pce_sampler to campaign
campaign.set_sampler(pce_sampler)

# Draw samples, execute and collate
campaign.execute(pool=client).collate(progress_bar=True)
campaign.execute()
samples = campaign.get_collation_result()


# %% [markdown]
# This will block until all evaluations are complete.
