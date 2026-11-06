import traceback

import pytest

from forge_idol.contracts import (
    CROP_MARGIN,
    MAX_GAUSSIANS,
    MIN_GAUSSIANS,
    IdolError,
    align_gaussians,
    crop_margin,
    raise_pending_error,
    set_pending_error,
)


def test_gaussians_stay_in_range_on_the_grid():
    assert align_gaussians(1) == MIN_GAUSSIANS
    assert align_gaussians(10 ** 9) == MAX_GAUSSIANS
    assert align_gaussians(262144 + 31) == 262144
    assert align_gaussians(262144 + 32) == 262176


def test_an_idol_error_prints_as_one_line():
    # what traceback.print_exc does from Python 3.12 on, which is what Forge's main thread calls
    try:
        raise IdolError("Add an input image")
    except IdolError as error:
        printed = "".join(traceback.format_exception(error))
    assert printed == "forge_idol.contracts.IdolError: Add an input image\n"


def test_other_errors_keep_their_traceback():
    try:
        raise ValueError("a real bug")
    except ValueError as error:
        printed = "".join(traceback.format_exception(error))
    assert printed.startswith("Traceback") and "test_contracts.py" in printed


def test_crop_margin_of_each_mode():
    assert crop_margin("synthetic") == CROP_MARGIN["synthetic"] == 1.2
    assert crop_margin("real") == 1.7
    with pytest.raises(IdolError, match="synthetic or real"):
        crop_margin("sketch")


def test_pending_error_stops_one_generation_only():
    # nothing clears it in txt2img, where the script does not run: a stale error must not answer for a later generation
    set_pending_error(IdolError("no"))
    with pytest.raises(IdolError):
        raise_pending_error()
    raise_pending_error()
