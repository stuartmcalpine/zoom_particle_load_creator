import argparse
import toml
import tempfile
import numpy as np
from particle_load import ParticleLoadParams
from pathlib import Path
from particle_load.Params import _compute_numbers, _check_params, _print_params, _load_zoom_mask
import pytest
from unittest.mock import patch, mock_open, MagicMock

@pytest.fixture
def mock_h5py_file():
    """Create a mock h5py file with the expected structure."""
    # Create the outer mock for the File object
    mock_file = MagicMock()
    
    # Create the mock that will be returned by __enter__
    mock_file_context = MagicMock()
    mock_file.__enter__.return_value = mock_file_context
    
    # Create the mock for the Coordinates group
    mock_coordinates = MagicMock()
    
    # Define the attribute values
    mock_coords_attrs = {
        "bounding_length": 10.0,
        "high_res_volume": 100.0,
        "grid_cell_width": 0.1,
        "geo_centre": np.array([50.0, 50.0, 50.0]),
    }
    
    # Set up the attributes dictionary
    mock_coordinates.attrs.get = lambda key: mock_coords_attrs.get(key)
    
    # Set up the coordinates data
    mock_coordinates.__getitem__.return_value = np.random.rand(100, 3)
    
    # Set up the return value for file access
    mock_file_context.__getitem__.return_value = mock_coordinates
    
    return mock_file

@pytest.fixture
def args():
    """Create a mock arguments object."""
    args = argparse.Namespace()
    args.param_file = "test_params.toml"
    args.make_ic_gen_param_files = True
    args.make_swift_param_files = False
    args.save_pl_data = False
    args.save_pl_data_hdf5 = False
    args.with_mpi = False
    args.make_extra_plots = False
    return args

# Test fixtures
@pytest.fixture
def minimal_params():
    """Create a minimal parameter dictionary with required fields."""
    return {
        "parent": {
            "box_size": 100.0,  # Mpc/h
            "n_particles": 64**3,  # Must be a perfect cube of glass file N
        },
        "output": {
            "path": "/tmp/output",
        },
        "cosmology": {
            "Omega0": 1.0,
            "OmegaCDM": 0.85,
            "OmegaLambda": 0.7,
            "OmegaBaryon": 0.15,
            "HubbleParam": 0.7,
            "Sigma8": 0.8,
        },
        "glass_file": {
            "N": 8**3,  # Number of particles in glass file
        },
        "ic_gen": {
            "panphasian_descriptor": "[Panph1,L16,(7048,47296,60919),S1,CH75651445,2MPP_L340p5]",
            "linear_ps": "/path/to/powerspectrum.txt",
        },
        "softening": {
            "rule": "eagle",
        },
        "zoom": {
            "enable": False
        }
    }

# Tests for individual functions
def test_compute_numbers(minimal_params):
    """Test the _compute_numbers function."""
    # Make a copy of the params to avoid modifying the fixture
    params = dict(minimal_params)
    
    _compute_numbers(params)
    
    # Check derived values
    assert params["parent"]["N_glass_L"] == 8  # (64**3 / 8**3)**(1/3) = 8
    assert params["glass_file"]["L_mpch"] == 100.0 / 8  # box_size / N_glass_L
    assert params["ic_gen"]["ndim_fft_start"] == 1
    assert params["ic_gen"]["n_species"] == 1  # No zoom by default
    
    # Test with zoom enabled
    params["zoom"] = {"enable": True}
    _compute_numbers(params)
    assert params["ic_gen"]["n_species"] == 2

def test_check_params_valid(minimal_params):
    """Test _check_params with valid parameters."""
    _check_params(minimal_params)  # Should not raise

def test_check_params_invalid_particle_ratio(minimal_params):
    """Test _check_params with invalid particle to glass ratio."""
    params = dict(minimal_params)
    params["parent"]["n_particles"] = 65**3  # Not divisible by glass N
    
    with pytest.raises(ValueError, match="Number of particles must divide"):
        _check_params(params)

def test_check_params_invalid_softening_rule(minimal_params):
    """Test _check_params with invalid softening rule."""
    params = dict(minimal_params)
    params["softening"]["rule"] = "invalid_rule"

    with pytest.raises(ValueError, match="not allowed"):
        _check_params(params)

def test_check_params_missing_zoom_mask(minimal_params):
    """Test _check_params with zoom enabled but no mask."""
    params = dict(minimal_params)
    params["zoom"] = {"enable": True, "mask": None}

    with pytest.raises(ValueError, match="Must provide zoom mask"):
        _check_params(params)

def test_check_params_nonexistent_zoom_mask(minimal_params):
    """Test _check_params with zoom enabled but mask file doesn't exist."""
    params = dict(minimal_params)
    params["zoom"] = {"enable": True, "mask": "/nonexistent/mask.h5"}

    with pytest.raises(FileNotFoundError, match="Zoom mask file not found"):
        _check_params(params)

def test_print_params(minimal_params, capsys):
    """Test _print_params function output."""
    _print_params(minimal_params)

    captured = capsys.readouterr()
    assert "parent" in captured.out
    assert "box_size" in captured.out
    assert "100.0" in captured.out

def test_load_zoom_mask(minimal_params, mock_h5py_file):
    """Test _load_zoom_mask function."""
    params = dict(minimal_params)
    params["zoom"] = {"enable": True, "mask": "test_mask.h5"}

    with patch("h5py.File", return_value=mock_h5py_file) as mock_file:
        _load_zoom_mask(params)
        mock_file.assert_called_once_with(params["zoom"]["mask"], "r")

    # Check that attributes were loaded
    assert "bounding_length" in params["zoom"]
    assert params["zoom"]["bounding_length"] == 10.0
    assert "geo_centre" in params["zoom"]
    assert np.array_equal(params["zoom"]["geo_centre"], np.array([50.0, 50.0, 50.0]))
    assert "mask_coordinates" in params["zoom"]
    assert params["zoom"]["mask_coordinates"].shape[1] == 3  # Should be Nx3 array

