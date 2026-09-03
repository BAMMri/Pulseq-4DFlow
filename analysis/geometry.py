#!/usr/bin/env python3
# Copyright 2025. Marta Maggioni
# Based on code provided by: Moritz Blumenthal, Martin Schilling (Uecker Lab, University Medical Center Göttingen)

import numpy as np
import h5py
import ismrmrd
import xml.etree.ElementTree as ET

PCS_TRANSFORMATIONS = {
    "HFS": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    "HFP": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "FFS": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
}

ISMRMRD_NS = {'ismrmrd': 'http://www.ismrm.org/ISMRMRD'}

SIGN_CORRECTION = np.array([
    [-1, 1, 1, -1],
    [ 1,-1, 1, -1],
    [ 1, 1,-1,  1],
    [ 1, 1, 1,  1]
])


def affine(ori, offset=None, inverse=False):
    offset = np.zeros(3) if offset is None else np.array(offset)
    ori = np.array(ori)
    matrix = np.eye(4)
    if inverse:
        matrix[:3, :3] = ori.T
        matrix[:3, 3] = -ori.T @ offset
    else:
        matrix[:3, :3] = ori
        matrix[:3, 3] = offset
    return matrix


class Geometry:
    def from_ismrmrd(self, dset, xml_source):
        self.acq = dset.read_acquisition(0)

        root = self._parse_xml(xml_source)

        self.patient_position = root.find(
            './/ismrmrd:measurementInformation/ismrmrd:patientPosition', ISMRMRD_NS
        ).text

        recon_space = root.find('.//ismrmrd:encoding/ismrmrd:reconSpace', ISMRMRD_NS)
        matrix_elem = recon_space.find('ismrmrd:matrixSize', ISMRMRD_NS)
        fov_elem = recon_space.find('ismrmrd:fieldOfView_mm', ISMRMRD_NS)
        table_elem = root.find('.//ismrmrd:relativeTablePosition', ISMRMRD_NS)

        # reconSpace y stores full k-space lines; convert to image matrix size
        self.matrix_size = [
            int(matrix_elem.find('ismrmrd:x', ISMRMRD_NS).text),
            int(np.ceil(int(matrix_elem.find('ismrmrd:y', ISMRMRD_NS).text) / 2)),
            int(matrix_elem.find('ismrmrd:z', ISMRMRD_NS).text),
        ]

        self.fov = [
            float(fov_elem.find('ismrmrd:x', ISMRMRD_NS).text),
            float(fov_elem.find('ismrmrd:y', ISMRMRD_NS).text),
            float(fov_elem.find('ismrmrd:z', ISMRMRD_NS).text),
        ]

        self.table_position = [
            int(table_elem.find('ismrmrd:x', ISMRMRD_NS).text),
            int(table_elem.find('ismrmrd:y', ISMRMRD_NS).text),
            int(table_elem.find('ismrmrd:z', ISMRMRD_NS).text),
        ]

        self.prs = self._calc_prs()

    def _parse_xml(self, xml_source):
        if isinstance(xml_source, str):
            return ET.parse(xml_source).getroot()
        elif isinstance(xml_source, (np.ndarray, bytes)):
            xml_str = xml_source.tobytes().decode('latin-1') if isinstance(xml_source, np.ndarray) else xml_source.decode('latin-1')
            return ET.fromstring(xml_str)
        raise ValueError("xml_source must be a file path (str) or numpy byte array")

    def _calc_prs(self):
        gp = np.array(self.acq.phase_dir)
        gr = np.array(self.acq.read_dir)
        gs = np.array(self.acq.slice_dir)

        pixel_sizes = [
            self.fov[1] / self.matrix_size[1],  # phase
            self.fov[0] / self.matrix_size[0],  # read
            self.fov[2] / self.matrix_size[2],  # slice
        ]

        self.pixel_size = pixel_sizes
        ori = np.stack([gp, gr, gs]).T @ np.diag(pixel_sizes)
        return affine(ori, self.acq.position)

    def get_dcm(self):
        t = PCS_TRANSFORMATIONS.get(self.patient_position)
        if t is None:
            print(f"Warning: Unknown patient position '{self.patient_position}', assuming HFS")
            t = PCS_TRANSFORMATIONS["HFS"]

        NR, NP, NS = self.matrix_size
        cen = np.array([-NP, NR, -NS]) / 2 + np.array([0, 0, 0.5])

        result = self.prs @ affine(t, cen)
        result[0:3, 3] += np.array(self.table_position)

        # Swap cols 0,1 and apply sign corrections to match DICOM convention
        result[:, [0, 1]] = result[:, [1, 0]]
        result[0, 2] = self.pixel_size[1]
        result[2, 0] = self.pixel_size[2]
        result *= SIGN_CORRECTION

        # --- Fix to make it openable with ITKSnap ---
        R_scaled = result[:3, :3]
        spacing = np.linalg.norm(R_scaled, axis=0)       # voxel spacing per axis
        R = R_scaled / spacing                            # unnormalized direction cosines
        # Nearest proper orthonormal rotation via SVD (polar decomposition)
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:                    # keep it a proper rotation (det = +1)
            U[:, -1] *= -1
            R_ortho = U @ Vt
        result_fixed = result.copy()
        result_fixed[:3, :3] = R_ortho * spacing          # reapply spacing
        print("max deviation fixed:", np.max(np.abs(result_fixed[:3, :3] - result[:3, :3])))
        result = result_fixed

        result[np.abs(result) < 1e-4] = 0.0
        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python geometry.py /path/to/scan.mrd")
        sys.exit(1)

    mrd_path = sys.argv[1]

    with h5py.File(mrd_path, 'r') as f:
        xml_bytes = np.array(f['dataset']['xml'][0])

    dset = ismrmrd.Dataset(mrd_path)
    geo = Geometry()
    geo.from_ismrmrd(dset, xml_bytes)
    dcm_matrix = geo.get_dcm()

    print("Column norms:")
    for i in range(3):
        print(f"  col{i}: {np.linalg.norm(dcm_matrix[:3, i]):.4f}")
    print("DICOM transform matrix:\n", dcm_matrix)
