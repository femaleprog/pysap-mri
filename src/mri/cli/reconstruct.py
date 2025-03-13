from hydra_zen import store, zen

from mri.io.output import save_data
from mri.cli.utils import raw_config, traj_config, setup_hydra_config, get_outdir_path
from mri.operators.fourier.utils import discard_frequency_outliers
from mrinufft.io.utils import add_phase_to_kspace_with_shifts, remove_extra_kspace_samples
from pymrt.recipes.coils import compress_svd
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.ggrappa import do_grappa_and_append_data, GRAPPA_RECON_AVAILABLE
from mrinufft.operators import FourierOperatorBase
import json
import numpy as np
import pickle as pkl
import logging
import os
import glob
from functools import partial
import pickle

import nibabel as nib
from scipy.ndimage import zoom
from mrinufft.operators.off_resonance import MRIFourierCorrected

log = logging.getLogger(__name__)

save_data_hydra = lambda x, * \
    args, **kwargs: save_data(get_outdir_path(x), *args, **kwargs)


def resample_b0_map(b0_map, target_shape):
    """Resample B0 map to match target image shape."""
    current_shape = b0_map.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return zoom(b0_map, zoom_factors, order=1)  # Linear interpolation


def dc_adjoint(obs_file: str | np.ndarray, traj_file: str, coil_compress: str | int, debug: int,
               obs_reader, traj_reader, fourier, grappa_recon=None, output_filename: str = "dc_adjoint.nii",
               return_data=False, orc: bool = False, b0_map_file: str = None):
    """
    Reconstructs an image using the adjoint operator with optional B0 off-resonance correction.
    """
    preprocessed_file = 'preprocessed_data.pkl'
    smaps_file = 'smaps.pkl'

    if os.path.exists(preprocessed_file):
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        kspace_data = preprocessed_data['kspace_data']
        kspace_loc = preprocessed_data['kspace_loc']
        data_header = preprocessed_data['data_header']
        traj_params = preprocessed_data['traj_params']
        shots = preprocessed_data['shots']
        log.info("Loaded preprocessed data from preprocessed_data.pkl")
    else:
        raw_data, data_header = obs_reader(obs_file)
        shots, traj_params = traj_reader(traj_file, dwell_time=traj_reader.keywords['raster_time'] /
                                         data_header["oversampling_factor"])
        traj_params['img_size'] = np.asarray([
            size + 1 if size % 2 else size for size in traj_params['img_size']
        ])
        kspace_data = np.squeeze(raw_data).astype(np.complex64)
        kspace_loc = shots.reshape(-1,
                                   traj_params["dimension"]).astype(np.float32)

        preprocessed_data = {
            'kspace_data': kspace_data,
            'kspace_loc': kspace_loc,
            'data_header': data_header,
            'traj_params': traj_params,
            'shots': shots
        }
        with open(preprocessed_file, 'wb') as f:
            pickle.dump(preprocessed_data, f)

    required_vars = {'kspace_data', 'kspace_loc',
                     'data_header', 'traj_params', 'shots'}
    if not all(var in preprocessed_data for var in required_vars):
        log.error(
            f"Missing required variables in preprocessed_data: {required_vars - set(preprocessed_data.keys())}")
        raise ValueError("Incomplete preprocessed data")

    smaps = None
    if os.path.exists(smaps_file):
        with open(smaps_file, 'rb') as f:
            smaps = pickle.load(f)
        log.info("Loaded smaps from smaps.pkl")
    else:
        fourier_op = fourier(
            kspace_loc,
            traj_params["img_size"],
            n_coils=data_header["n_coils"] if coil_compress == -
            1 else coil_compress,
        )
        log.debug("Forcing smaps computation with a dummy adj_op call")
        _ = fourier_op.adj_op(kspace_data)
        smaps = getattr(fourier_op, 'smaps', None)

        if smaps is not None:
            with open(smaps_file, 'wb') as f:
                pickle.dump(smaps, f)
            log.info("Saved computed smaps to smaps.pkl")
        else:
            log.error("Failed to compute smaps, they are None")
            raise ValueError("Computed smaps is None")

    fourier_op = fourier(
        kspace_loc,
        traj_params["img_size"],
        n_coils=data_header["n_coils"] if coil_compress == -
        1 else coil_compress,
        smaps=smaps,
    )

    if orc and b0_map_file:
        log.info("Applying B0 off-resonance correction with MRI Fourier Operator")
        b0_map_nii = nib.load(b0_map_file)
        b0_map = b0_map_nii.get_fdata().astype(np.float32)
        dwell_time = traj_reader.keywords['raster_time'] / \
            data_header["oversampling_factor"]
        readout_time = np.arange(kspace_loc.shape[0]) * dwell_time

    log.info("Getting the DC Adjoint")
    dc_adjoint = fourier_op.adj_op(kspace_data)
    if not getattr(fourier_op, 'uses_sense', False):
        dc_adjoint = np.linalg.norm(dc_adjoint, axis=0)

    log.info("Saving DC Adjoint")
    save_data_hydra(output_filename, dc_adjoint, data_header)
    if return_data:
        return dc_adjoint, (fourier_op, kspace_data, traj_params, data_header)


def recon(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str | int,
          algorithm: str, debug: int, obs_reader, traj_reader, fourier, linear, sparsity,
          output_filename: str = "recon.nii", remove_dc_for_recon: bool = True, validation_recon: np.ndarray = None, metrics: dict = None,
          grappa_recon=None):
    """Reconstructs an MRI image using the given parameters.

    Parameters
    ----------
    obs_file : str
        Path to the file containing the observed k-space data.
    traj_file : str
        Path to the file containing the trajectory data.
    mu : float
        Regularization parameter for the sparsity constraint.
    num_iterations : int
        Number of iterations for the reconstruction algorithm.
    coil_compress : str | int
        Method or factor for coil compression.
    algorithm : str
        Optimization algorithm to use for reconstruction.
    debug : int
        Debug level for printing debug information.
    obs_reader : callable
        Object for reading the observed k-space data.
    traj_reader : callable
        Object for reading the trajectory data.
    fourier : callable
        Object representing the Fourier operator.
    linear : callable
        Object representing the linear operator.
    sparsity : callable
        Object representing the sparsity operator.
    output_filename : str, optional
        Path to save the reconstructed image, by default "recon.pkl"
    remove_dc_for_recon: bool, optional
        Whether to remove the density compensation for reconstruction, by default True
        Note that it will still be used to estimate x_init
    validation_recon: np.ndarray, optional
        The validation reconstruction to compare the results with, by default None
    metrics: dict, optional
        List of metrics to evaluate the reconstruction, by default None
    """
    recon_adjoint, additional_data = dc_adjoint(
        obs_file,
        traj_file,
        coil_compress,
        debug,
        obs_reader,
        traj_reader,
        fourier,
        grappa_recon=grappa_recon,
        output_filename='dc_adj_' + output_filename,
        return_data=True,
    )
    fourier_op, kspace_data, traj_params, data_header = additional_data
    if remove_dc_for_recon:
        fourier_op.impl.density = None
    K = fourier_op.op(recon_adjoint)
    alpha = np.mean(np.linalg.norm(kspace_data, axis=0)) / \
        np.mean(np.linalg.norm(K, axis=0))
    recon_adjoint *= alpha
    linear_op = linear(shape=tuple(
        traj_params["img_size"]), dim=traj_params['dimension'])
    linear_op.op(recon_adjoint)
    sparse_op = sparsity(coeffs_shape=linear_op.coeffs_shape, weights=mu)
    log.info("Setting up reconstructor")
    reconstructor = SelfCalibrationReconstructor(
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=sparse_op,
        verbose=1,
        lipschitz_cst=fourier_op.impl.get_lipschitz_cst(),
    )
    log.info("Starting reconstruction")
    recon, costs, metrics_iter = reconstructor.reconstruct(
        kspace_data=kspace_data,
        optimization_alg=algorithm,
        x_init=recon_adjoint,  # gain back the first step by initializing with DC Adjoint
        num_iterations=num_iterations,
    )
    if validation_recon is not None:
        log.info("getting metrics of the reconstruction")
        final_metrics = {}
        for metric, function in metrics.items():
            final_metrics[metric] = function(recon, validation_recon)
            final_metrics[f"dc_{metric}"] = function(
                recon_adjoint, validation_recon)
        log.info(f"Final Metrics: {final_metrics}")
        with open(get_outdir_path('metrics.json'), 'w') as f:
            final_metrics["traj"] = data_header["trajectory_name"]
            f.write(json.dumps(final_metrics, indent=4))
        data_header['metrics'] = final_metrics
    data_header['costs'] = costs
    data_header['metrics_iter'] = metrics_iter
    log.info("Saving reconstruction results")
    save_data_hydra(output_filename, recon, data_header)


setup_hydra_config()
store(
    dc_adjoint,
    obs_reader=raw_config,
    traj_reader=traj_config,
    coil_compress=10,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "disable"},
        {"fourier/smaps": "low_frequency"},
    ],
    name="dc_adjoint",
)
store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=30,
    coil_compress=10,
    mu=1e-7,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "disable"},
        {"fourier/smaps": "low_frequency"},
        {"linear": "gpu"},
        {"sparsity": "weighted_sparse"},
    ],
    name="recon",
)

store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=30,
    coil_compress=5,
    mu=1e-7,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu_lowmem"},
        {"grappa_recon": "disable"},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "low_frequency"},
    ],
    name="recon_lowmem",
)

# Setup the Hydra Config and callbacks.
store.add_to_hydra_store()


def run_recon():
    zen(recon).hydra_main(
        config_name="recon",
        config_path=None,
        version_base="1.3",
    )


def run_adjoint():
    zen(dc_adjoint).hydra_main(
        config_name="dc_adjoint",
        config_path=None,
        version_base="1.3",
    )
