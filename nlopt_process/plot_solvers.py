"""Define functions to run various regression test input files using various
solvers, and compare the results.
"""
import process
from process.main import SingleRun
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from dataclasses import dataclass

PROCESS_SCENARIOS_DIR = (
    Path(process.__file__).parent.parent / "tests/regression/scenarios/"
).resolve()
# TODO Should be made in nb; check if exists, make if not
DATA_DIR = Path("data").resolve()

# Translate figure of merit number to variable name
FOM_DESCRIPTIONS = [
    "major radius",
    "not used",
    "neutron wall load",
    "P_tf + P_pf",
    "fusion gain",
    "cost of electricity",
    "capital cost",
    "aspect ratio",
    "divertor heat load",
    "toroidal field",
    "total injected power",
    "H plant capital cost",
    "H production rate",
    "pulse length",
    "plant availability",
    "min R0, max tau_burn",
    "net electrical outpu.",
    "Null figure of merit",
    "max Q, max t_burn",
]
NORM_OPT_PARAM_NAMES = "xcm|norm_objf"

# TODO Use parent logging once in Process
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
s_handler = logging.StreamHandler()
s_handler.setLevel(logging.DEBUG)
logger.addHandler(s_handler)


@dataclass
class RunMetadata:
    """Metadata for a given Process run.

    Define mfile for run and other information to identify it. Depending on
    what is being run/plotted, different fields can be undefined.
    """

    mfile_path: Path
    tag: str
    scenario: str = ""
    solver: str = ""


def copy_regression_ref(scenario):
    """Copy regression reference from Process to notebook's data dir.

    :param scenario: name of regression scenario
    :type scenario: str
    :return: path to reference mfile in data dir
    :rtype: pathlib.Path
    """
    proc_ref_mfile_path = PROCESS_SCENARIOS_DIR / scenario / "ref.MFILE.DAT"
    new_ref_mfile_path = DATA_DIR / (scenario + "_ref_MFILE.DAT")
    shutil.copy(proc_ref_mfile_path, new_ref_mfile_path)

    return new_ref_mfile_path


# TODO Could make solver opt
def run_regression_input(scenario, solver):
    """Copy and run a regression input file using a given solver.

    :param scenario: regression test to run
    :type scenario: str
    :param solver: solver to use (e.g. nlopt)
    :type solver: str
    :return: path to output mfile
    :rtype: pathlib.Path
    """
    # Define Process input file path, then copy to CWD
    proc_input_file_path = PROCESS_SCENARIOS_DIR / scenario / "IN.DAT"
    new_input_path = DATA_DIR / f"{scenario}_{solver}_IN.DAT"
    shutil.copy(proc_input_file_path, new_input_path)

    # Run process on the input file in the data dir, using solver of choice
    # TODO Make Process able to import main?
    try:
        single_run = process.main.SingleRun(str(new_input_path), solver=solver)
    except:
        logger.exception("Running input file failed.")
        raise

    mfile_path = _get_mfile_path(scenario, solver)

    return mfile_path


def plot_mfile_solutions(
    sol1_path, sol2_path, sol1_tag, sol2_tag, title, percentage=True
):
    """Manually plot two arbitrary mfiles, together or as a percentage difference.

    Plots the final optimisation parameters for two mfiles on the same plot.
    Alternatively, plots the percentage difference between them.

    This is used when two arbitrary mfiles need comparing.
    :param sol1_path: path to solution 1's mfile
    :type sol1_path: pathlib.Path
    :param sol2_path: path to solution 2's mfile
    :type sol2_path: pathlib.Path
    :param sol1_tag: plot label for solution 1
    :type sol1_tag: str
    :param sol2_tag: plot label for solution 2
    :type sol2_tag: str
    :param title: title of plot
    :type title: str
    :param percentage: percentage difference plot, defaults to True
    :type percentage: bool, optional
    """
    # Create run metadata for the two mfiles
    runs_metadata = [
        RunMetadata(mfile_path=sol1_path, tag=sol1_tag),
        RunMetadata(mfile_path=sol2_path, tag=sol2_tag),
    ]

    # Parse mfiles and put into dataframe
    results_df = _create_df_from_run_metadata(runs_metadata)

    if percentage:
        # Calculate percentage differences on optimisation parameters and FOM
        diffs_df = _percentage_changes(results_df, sol1_tag)
        _plot_solution_diffs(diffs_df, title)
    else:
        _plot_two_solutions(results_df, sol1_tag, sol2_tag, title)


def _create_runs_metadata(scenarios, solvers):
    """Create metadata for each run to be performed.

    :param scenarios: regression scenarios
    :type scenarios: list
    :param solvers: solvers to use
    :type solvers: list
    :return: RunMetadata objects for each run
    :rtype: list[RunMetadata]
    """
    runs_metadata = []
    for solver in solvers:
        for scenario in scenarios:
            # TODO Possibly only set mfile path if it exists? (run or found)
            mfile_path = _get_mfile_path(scenario, solver)

            run_metadata = RunMetadata(
                mfile_path=mfile_path,
                tag=f"{scenario}_{solver}",
                scenario=scenario,
                solver=solver,
            )
            runs_metadata.append(run_metadata)

    return runs_metadata


def _run_scenarios(runs_metadata, rerun_tests):
    """Run scenarios based on the metadata list.

    :param runs_metadata: scenarios and solvers to run
    :type runs_metadata: list[RunMetadata]
    """
    for run_metadata in runs_metadata:
        # TODO Might need to catch errors/failures
        # Don't re-run if mfile already exists and rerun isn't forced
        if run_metadata.mfile_path.exists() and rerun_tests == False:
            continue
        else:
            try:
                scenario = run_metadata.scenario
                solver = run_metadata.solver
                run_regression_input(scenario, solver)
            except RuntimeError:
                logger.warn(f"{scenario} using {solver} has failed.")


def _get_mfile_path(scenario, solver):
    """Create mfile path from scenario and solver strings.

    :param scenario: scenario being run
    :type scenario: str
    :param solver: solver being used
    :type solver: str
    :return: output mfile path
    :rtype: pathlib.Path
    """
    run_path_prefix = scenario + "_" + solver
    mfile_path = DATA_DIR / (run_path_prefix + "_MFILE.DAT")
    return mfile_path


def _extract_mfile_data(mfile_path):
    """Extract data from mfile and return dict.

    :param mfile_path: mfile to extract data from
    :type mfile_path: pathlib.Path
    :return: dict of all data in mfile
    :rtype: dict
    """
    mfile = process.io.mfile.MFile(str(mfile_path))
    mfile_data = {}

    for var in mfile.data.keys():
        mfile_data[var] = mfile.data[var].get_scan(-1)

    return mfile_data


def _create_df_from_run_metadata(runs_metadata):
    """Create a dataframe from multiple mfiles.

    Uses RunMetadata objects.
    :param runs_metadata: scenarios and solvers that have been run
    :type runs_metadata: list[RunMetadata]
    :return: dataframe of all results
    :rtype: pandas.DataFrame
    """
    # TODO Need to reconcile this with above func
    # Need to write tests for previous functionality first
    results = []
    for run_metadata in runs_metadata:
        if Path(run_metadata.mfile_path).exists():
            mfile_data = _extract_mfile_data(run_metadata.mfile_path)
        else:
            raise FileNotFoundError(
                f"The MFILE {run_metadata.mfile_path} doesn't exist"
            )
        # Merge each run's metadata and results into one dict
        results.append({**run_metadata.__dict__, **mfile_data})

    results_df = pd.DataFrame(results)
    return results_df


# TODO Still need a way to get objf label from minmax int value
# Not called
def _extract_foms(results_df):
    """Determine the figure of merit descriptions.

    Makes plotting the figure of merit for each scenario much easier.
    :param results_df: run results
    :type results_df: pandas.DataFrame
    """
    # Determine objective function description from minmax
    results_df["FOM var name"] = results_df["minmax"].apply(
        lambda minmax: FOM_DESCRIPTIONS[int(abs(minmax)) - 1]
    )


def _percentage_changes(results_df, normalising_tag):
    """Percentage differences between different solvers running the same scenario.

    Percentage diffs of multiple solutions relative to one.
    :param results_df: dataframe of two solutions (same scenario, different solvers)
    :type results_df: pandas.DataFrame
    :return: percentage differences
    :rtype: pandas.DataFrame
    """
    # Calculate the percentage differences between multiple solutions,
    # for optimisation parameters and FOM
    is_opt_param = results_df.columns.str.contains(NORM_OPT_PARAM_NAMES)
    tags = results_df["tag"]
    normalising_soln = results_df.loc[
        results_df["tag"] == normalising_tag, is_opt_param
    ]
    normalising_soln_np = normalising_soln.to_numpy()

    # Solutions that need percentage diffs calculating
    solns = results_df.loc[results_df["tag"] != normalising_tag, is_opt_param]

    # Calculate percentage diffs
    diffs_df = ((solns - normalising_soln_np) / abs(normalising_soln_np)) * 100

    # Combine dfs to get tag alongside percentage diff
    tags_minus_norm_tag = results_df.loc[results_df["tag"] != normalising_tag, "tag"]
    diffs_df = pd.concat([tags_minus_norm_tag, diffs_df], axis=1)

    return diffs_df


# Main plotting func
def compare_regression_test(scenario, solvers, rerun_tests=False):
    """Run a single regression test for two solvers and return differences.

    The percentage differences between the two solutions will be returned; the
    first solver's solution being used as the normalising solution.
    :param scenario: regression test to run
    :type scenario: str
    :param solvers: the two solvers to run with
    :type solvers: list[str]
    :param rerun_tests: re-run tests or use existing results, defaults to False
    :type rerun_tests: bool, optional
    :return: dataframe of results
    :rtype: pandas.DataFrame
    """
    assert type(scenario) is str
    assert type(solvers) is list and len(solvers) == 2

    # Create run metadata, run tests if results non-existent, re-run tests if
    # forced
    runs_metadata = _create_runs_metadata([scenario], solvers)
    _run_scenarios(runs_metadata, rerun_tests)

    # Parse mfiles and put into dataframe
    results_df = _create_df_from_run_metadata(runs_metadata)

    # Calculate percentage differences on optimisation parameters and FOM,
    # relative to the first solver's solution
    normalising_tag = results_df.loc[results_df["solver"] == solvers[0], "tag"].values[
        0
    ]
    diffs_df = _percentage_changes(results_df, normalising_tag)

    title = (
        f"Percentage differences between {solvers[0]} and \n{solvers[1]} "
        f"for {scenario}"
    )
    _plot_solution_diffs(diffs_df, title)

    return diffs_df


# Main plotting func
# TODO Some similarities with compare_regression_test(); DRY? Different enough I think
def compare_regression_tests(solvers, rerun_tests=False):
    """Run all regression tests for two solvers and plot differences.

    :param solvers: two solvers to compare
    :type solvers: list[str]
    :param rerun_tests: force re-running of tests, defaults to False
    :type rerun_tests: bool, optional
    :return: RMS errors for solutions for both solvers
    :rtype: pandas.DataFrame
    """
    assert type(solvers) is list and len(solvers) == 2

    # Run all input files with VMCON and solver to test
    scenario_dirs = PROCESS_SCENARIOS_DIR.iterdir()
    scenarios = [scenario_dir.stem for scenario_dir in scenario_dirs]

    # Remove unwanted tests
    scenarios.remove("IFE")  # uses HYBRD
    scenarios.remove("stellarator_config")  # crashes

    runs_metadata = _create_runs_metadata(scenarios, solvers)
    _run_scenarios(runs_metadata, rerun_tests)

    # Parse mfiles and put into dataframe
    results_df = _create_df_from_run_metadata(runs_metadata)

    filtered_results_df = _filter_vars_of_interest(
        results_df, ["scenario", "solver", "minmax", "tag"]
    )

    # Calculate RMS error and plot
    rms_error_df = rms_error(filtered_results_df)
    _plot_rmses(rms_error_df, solvers)

    clear_unwanted_output()

    return rms_error_df


def _filter_vars_of_interest(results_df, var_names=None):
    """Filter variables of interest from full results for all solutions.

    :param results_df: full results for all solutions
    :type results_df: pandas.DataFrame
    :param var_names: variables of interest to filter, defaults to None
    :type var_names: list[str], optional
    :return: variables of interest
    :rtype: pandas.DataFrame
    """
    # Filter in tag by default
    if var_names is None:
        var_names = ["tag"]

    # Filter for opt params
    opt_params = results_df.loc[
        :, results_df.columns.str.contains(NORM_OPT_PARAM_NAMES)
    ]
    vars_of_interest = results_df[var_names]
    # Concatenate filtered columns
    filtered_results = pd.concat([vars_of_interest, opt_params], axis=1)

    return filtered_results


def rms_error(results_df):
    """Calculate RMS errors between scenario solutions using different solvers.

    :param results_df: solutions for all scenarios and solvers
    :type results_df: pandas.DataFrame
    :return: RMS errors for different scenarios
    :rtype: pandas.DataFrame
    """
    # Filter out optimisation parameters only (not norm_objf as well, as in
    # NORM_OPT_PARAM_NAMES)
    is_opt_param = results_df.columns.str.contains("xcm")

    # Filter out scenario runs that didn't converge for both solvers
    # (use minmax = NaN as a sign of not working)
    didnt_conv = results_df[results_df["minmax"].isna()]
    conv_results_df = results_df[
        ~results_df["scenario"].isin(didnt_conv["scenario"].unique())
    ]

    # Find pairs of results for the same scenario, different solvers
    # TODO Iterating over dataframe is bad: better solution?
    scenarios = conv_results_df["scenario"].unique()
    solvers = conv_results_df["solver"].unique()
    rmses = []
    for scenario in scenarios:
        scenario_results = conv_results_df[conv_results_df["scenario"] == scenario]

        # For a given scenario, get optimisation parameters for each solver run
        x1_df = scenario_results.loc[
            scenario_results["solver"] == solvers[0], is_opt_param
        ]
        x2_df = scenario_results.loc[
            scenario_results["solver"] == solvers[1], is_opt_param
        ]

        # Convert to numpy array (avoid resetting df indexes) and flatten
        x1 = x1_df.to_numpy().ravel()
        x2 = x2_df.to_numpy().ravel()

        # Drop NaN values: different scenarios have different numbers of
        # optimisation parameters
        x1 = x1[~np.isnan(x1)]
        x2 = x2[~np.isnan(x2)]

        rmse = np.sqrt(np.mean((x2 - x1) ** 2))
        rmses.append(rmse)

    rmse_df = pd.DataFrame([rmses], columns=scenarios)
    return rmse_df


def _plot_solution_diffs(diffs_df, plot_title):
    """Plot percentage differences between multiple solutions to a given scenario.

    :param diffs_df: percentage diffs for iterations vars and FOM
    :type diffs_df: pandas.DataFrame
    :param plot_title: title of plot
    :type plot_title: str
    """
    # Melt df (wide to long-form) for seaborn plotting with jitter
    diffs_df = diffs_df.melt(id_vars="tag")

    # Separate optimisation parameters and objective dfs
    opt_params_df = diffs_df[diffs_df["variable"] != "norm_objf"]
    norm_objf_df = diffs_df[diffs_df["variable"] == "norm_objf"]

    # Separate optimisation parameters and objective function subplots
    fig, ax = plt.subplots(ncols=2, width_ratios=[5, 1])
    fig.suptitle(plot_title)

    # Strip plot for optimisation parameters
    sns.stripplot(
        data=opt_params_df,
        x="variable",
        y="value",
        hue="tag",
        jitter=True,
        ax=ax[0],
    )

    ax[0].set_ylabel("Percentage difference")
    ax[0].set_xlabel("Optimisation parameter")
    ax[0].legend()
    opt_params_labels = ax[0].get_xticklabels()

    for label in opt_params_labels:
        fmt_label = label.get_text().lstrip("xcm0")
        label.set_text(fmt_label)

    ax[0].set_xticklabels(opt_params_labels)

    # Hide every other x-tick
    for label in opt_params_labels[::2]:
        label.set_visible(False)

    # Plot objf change separately
    sns.stripplot(
        data=norm_objf_df,
        x="variable",
        y="value",
        hue="tag",
        jitter=True,
        ax=ax[1],
    )

    norm_objf_label = ax[1].get_xticklabels()
    for label in norm_objf_label:
        fmt_label = label.get_text().replace("norm_", "")
        label.set_text(fmt_label)

    ax[1].set_xticklabels(norm_objf_label)
    ax[1].get_legend().remove()
    ax[1].set_ylabel("")
    # TODO Use actual objective function label
    ax[1].set_xlabel("Objective")

    # Ensure title doesn't overlap plots
    fig.tight_layout()


# TODO Could be made more generic i.e. multiple solns
def _plot_two_solutions(results_df, sol1_label, sol2_label, title):
    """Plot optimisation parameters for two solutions together.

    :param results_df: results for two solutions
    :type results_df: pandas.DataFrame
    :param sol1_label: label for solution 1
    :type sol1_label: str
    :param sol2_label: label for solution 2
    :type sol2_label: str
    :param title: title for plot
    :type title: str
    """
    # Use normalised values when comparing 2 solutions (e.g. xcm001)
    # FOM is tricky to normalise (no initial value), so normalise to first
    # solution
    results_df["Normalised FOM"] = (
        results_df["norm_objf"] / results_df.loc[0, "norm_objf"]
    )
    is_opt_param = results_df.columns.str.contains(NORM_OPT_PARAM_NAMES)
    opt_param_labels = results_df.columns.str.lstrip("xcm0")[is_opt_param]
    opt_param_labels = opt_param_labels.str.lstrip("Normalised")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(
        opt_param_labels,
        results_df.loc[0, is_opt_param],
        "o",
        label=sol1_label,
    )
    ax.plot(
        opt_param_labels,
        results_df.loc[1, is_opt_param],
        "o",
        label=sol2_label,
    )
    ax.set_ylabel("Normalised value")
    ax.set_xlabel("Iteration variable and FOM")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)


def _plot_rmses(rmses_df, solvers):
    """Plot RMSEs for scenarios run with two different solvers.

    :param rmses_df: RMSEs for scenarios with different solvers
    :type rmses_df: pandas.DataFrame
    :param solvers: the two solvers to compare
    :type solvers: list[str]
    """
    fig, ax = plt.subplots()
    scenario_names = [name[:10] for name in rmses_df.columns]
    ax.plot(scenario_names, rmses_df.loc[0, :], "o")
    # ax.set_xlabel("Regression test")
    ax.set_ylabel("RMS error")
    ax.set_title(
        f"RMS errors between {solvers[0]} and \n{solvers[1]}'s "
        "solutions for various scenarios"
    )
    plt.xticks(rotation=60)


def clear_unwanted_output():
    """Remove unwanted files from run."""
    unwanted_types = ["*OPT.DAT", "*PLOT.DAT", "*SIG_TF.DAT"]
    to_deletes = []
    for unwanted_type in unwanted_types:
        to_deletes.extend(list(DATA_DIR.glob(unwanted_type)))

    for to_delete in to_deletes:
        to_delete.unlink()


# Main func
# Could be the generic case
def global_comparison(runs_metadata, relative_tag, plot_title):
    """Plot multiple solutions relative to a reference solution.

    :param runs_metadata: list of RunMetadata objects
    :type runs_metadata: list[RunMetadata]
    :param relative_tag: tagged run to compare against
    :type relative_tag: str
    :param plot_title: title of plot
    :type plot_title: str
    """
    # Create dataframe from runs metadata: mfile data with a tag for each run
    results_df = _create_df_from_run_metadata(runs_metadata)

    # Filter for tag, optimisation parameters and objective function
    filtered_results_df = _filter_vars_of_interest(results_df)

    # Work out the percentage diffs relative to a certain tagged solution
    diffs_df = _percentage_changes(filtered_results_df, relative_tag)

    # Plot
    _plot_solution_diffs(diffs_df, plot_title)

    return results_df
