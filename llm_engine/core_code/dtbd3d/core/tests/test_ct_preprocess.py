from __future__ import annotations

import numpy as np
import nibabel as nib

from dtbd3d.core.ct_preprocess import nifti_to_hu_array


def test_nifti_hu_array_does_not_apply_metadata_intercept() -> None:
    image = nib.Nifti1Image(np.array([[[-1024, 0, 3071]]], dtype=np.int16), affine=np.eye(4))

    hu = nifti_to_hu_array(image)

    assert hu.dtype == np.float32
    assert hu.min() == -1024.0
    assert hu.max() == 3071.0
