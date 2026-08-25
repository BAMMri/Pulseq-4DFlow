#!/usr/bin/env python3
import os
import json
import subprocess
import numpy as np
import sys
import h5py
import ismrmrd
from geometry import Geometry
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

# Set GPU environment variables for BART
os.environ['BART_CUDA_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0, change if you have multiple GPUs

import bart
from bart import cfl

bart_orig = bart.bart

def bart_print(nargout, cmd, *args):
    print("Executing bart", cmd)
    return bart_orig(nargout, cmd, *args)


bart.bart = bart_print


def reconstruct_and_process_all(data_dir, venc_cms=20, use_gpu=True, ecalib_r='20:20', compressed_coils=0,
                                joint_recon=False):
    """Reconstruct all .cfl/.hdr pairs in data_dir and return velocity results.

    Returns a dict keyed by file_base with entries:
        {'velocities': ndarray (x,y,z,t,3) in m/s, 'mask': ndarray, 'magnitude': ndarray}
    """
    print("\n[2/3]Performing recon...")
    if use_gpu:
        print("GPU acceleration enabled")
        os.environ['BART_CUDA_GPU'] = '1'
    else:
        print("GPU acceleration disabled")
        os.environ['BART_CUDA_GPU'] = '0'

    print(f'{ecalib_r=}')

    file_bases = [
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.endswith('.cfl') and os.path.exists(os.path.join(data_dir, os.path.splitext(f)[0] + '.hdr'))
    ]
    if not file_bases:
        print("No .cfl/.hdr data files found in the directory. Exiting.")
        return {}

    results = {}
    for file_base in file_bases:
        cfl_path = os.path.join(data_dir, file_base)
        print(f"Reading: {cfl_path}")
        ksp = cfl.readcfl(cfl_path)

        if compressed_coils > 0:
            ksp = bart.bart(1, f'cc -p{compressed_coils}', ksp)

        num_x, num_y, num_slices = ksp.shape[0:3]
        num_coils = ksp.shape[3]
        num_cardiac_phases = ksp.shape[6]
        num_vencs = ksp.shape[11]
        print(f"Shape: {ksp.shape}, num_vencs: {num_vencs}, num_cardiac_phases: {num_cardiac_phases}")

        recon = np.zeros((num_x, num_y, num_slices, num_cardiac_phases, num_vencs), dtype=np.complex64)
        print('venc_cms', venc_cms)

        def recon_separate():
            for venc in range(num_vencs):
                for phase in range(num_cardiac_phases):
                    print(f"Processing venc {venc + 1}/{num_vencs}, phase {phase + 1}/{num_cardiac_phases}")
                    all_ksp_venc_phase = ksp[:, :, :, :, 0, 0, phase, 0, 0, 0, 0, venc]
                    all_ksp_esprit = ksp[:, :, :, :, 0, 0, phase, 0, 0, 0, 0, 0]

                    if use_gpu:
                        sensitivities = bart.bart(1, f"ecalib -g -c0 -m1 -r{ecalib_r}", all_ksp_esprit)
                    else:
                        sensitivities = bart.bart(1, f"ecalib -c0 -m1 -r{ecalib_r}", all_ksp_esprit)

                    l1_wav_reg = 0.005
                    if use_gpu:
                        image_l1_tv_wav = bart.bart(
                            1, f"pics -g -R W:7:0:{l1_wav_reg} -e -i 20 -S -d5",
                            all_ksp_venc_phase, sensitivities)
                    else:
                        image_l1_tv_wav = bart.bart(
                            1, f"pics -R W:7:0:{l1_wav_reg} -e -i 20 -S -d5",
                            all_ksp_venc_phase, sensitivities)

                    recon[:, :, :, phase, venc] = image_l1_tv_wav.squeeze()

        def recon_joint():
            nonlocal recon
            ksp_flattened = np.reshape(
                ksp, (num_x, num_y, num_slices, num_coils, 1, 1, num_cardiac_phases * num_vencs, 1, 1, 1, 1, 1))
            ksp_espirit = np.sum(ksp_flattened, axis=6)
            gpu_flag = '-g' if use_gpu else ''
            sensitivities = bart.bart(1, f"ecalib {gpu_flag} -c0 -m1 -r{ecalib_r}", ksp_espirit)

            for venc in tqdm(range(num_vencs)):
                l1_wav_reg = 0.009
                tv_reg = 0.0001
                ksp_to_recon = np.zeros(
                    (num_x, num_y, num_slices, num_coils, 1, 1, num_cardiac_phases, 1, 1, 1, 1, 1), np.complex64)
                ksp_to_recon[:, :, :, :, 0, 0, :, 0, 0, 0, 0, 0] = ksp[:, :, :, :, 0, 0, :, 0, 0, 0, 0, venc]
                print("Shape to recon", ksp_to_recon.shape)
                image_l1_tv_wav = bart.bart(
                    1, f"pics {gpu_flag} -R W:7:0:{l1_wav_reg} -R T:64:0:{tv_reg} -e -i 20 -S -d2",
                    ksp_to_recon, sensitivities)
                recon[:, :, :, :, venc] = image_l1_tv_wav.squeeze()

        if joint_recon:
            recon_joint()
        else:
            recon_separate()

        if recon.shape[-1] < 4:
            print(f"ERROR: Not enough VENC encodings to compute x, y, z velocities for {file_base}!")
            continue

        phase_diff_x = np.angle(recon[..., 1] * np.conj(recon[..., 0]))
        phase_diff_y = np.angle(recon[..., 2] * np.conj(recon[..., 0]))
        phase_diff_z = np.angle(recon[..., 3] * np.conj(recon[..., 0]))
        phase_diffs = np.stack([phase_diff_x, phase_diff_y, phase_diff_z], axis=-1)
        
        # Compute velocities in m/s (ORMIR-MIDS requires m/s)
        velocities_ms = (phase_diffs / np.pi) * (venc_cms / 100.0)
        for axis in range(3):  # x, y, z
        
            temporal_mean = np.mean(velocities_ms[:, :, :, :, axis], axis=3, keepdims=True)
            # Subtract voxel-wise temporal mean from every phase
            velocities_ms[:, :, :, :, axis] -= temporal_mean
        
        mag_data = np.abs(recon[..., 0])
        mask = mag_data > 0.10 * np.max(mag_data)
        velocities_ms = velocities_ms * mask[..., np.newaxis]

        results[file_base] = {
            'velocities': velocities_ms,
            'mask': mask,
            'magnitude': np.abs(recon),
        }
        print(f"Processed {file_base}: velocities shape {velocities_ms.shape}")

    return results


def extract_geometry(dataset: ismrmrd.Dataset, mrd_path: Path) -> np.ndarray:
    """Extract the 4x4 affine matrix from an open ismrmrd.Dataset. Returns np.eye(4) on failure."""
    print("Extracting geometry...")
    try:
        with h5py.File(mrd_path, 'r') as f:
            xml_bytes = np.array(f['dataset']['xml'][0])
        geo = Geometry()
        geo.from_ismrmrd(dataset, xml_bytes)
        return geo.get_dcm()
    except Exception as e:
        print(f"  WARNING: Could not extract geometry ({e}). Using identity affine.")
        return np.eye(4)


def convert_mrd_to_bart(dataset: ismrmrd.Dataset, output_dir: Path, stem: str) -> Path:
    print("\n[1/3] Converting .mrd -> BART format...")
    cfl_path = output_dir / (stem + '.cfl')
    hdr_path = output_dir / (stem + '.hdr')

    if cfl_path.exists() and hdr_path.exists():
        print("  BART files exist, skipping conversion.")
        return output_dir

    # Single bulk HDF5 read — vastly faster than N individual read_acquisition() calls
    raw = dataset._dataset['data'][:]
    heads = raw['head']
    idx_fields = heads['idx']

    acq0 = ismrmrd.Acquisition(heads[0])
    channels, matrix_x = acq0.active_channels, acq0.number_of_samples

    # Vectorized dimension discovery (one numpy pass instead of N Python iterations)
    m_y  = int(idx_fields['kspace_encode_step_1'].max()) + 1
    m_z  = int(idx_fields['kspace_encode_step_2'].max()) + 1
    c    = int(idx_fields['contrast'].max()) + 1
    ph   = int(idx_fields['phase'].max()) + 1
    r    = int(idx_fields['repetition'].max()) + 1
    s    = int(idx_fields['set'].max()) + 1
    seg  = int(idx_fields['segment'].max()) + 1
    sl   = int(idx_fields['slice'].max()) + 1
    avg  = int(idx_fields['average'].max()) + 1

    out_shape = (matrix_x, m_y, m_z, channels, 1, c, ph, 1, 1, 1, r, s, seg, sl, avg, 1)
    output_mmap = np.memmap(str(cfl_path), dtype='<c8', mode='w+', shape=out_shape, order='F')

    for i in tqdm(range(len(raw))):
        ic = idx_fields[i]
        data = raw[i]['data'].view(np.complex64).reshape((channels, matrix_x))
        output_mmap[:, ic['kspace_encode_step_1'], ic['kspace_encode_step_2'], :, 0,
                    ic['contrast'], ic['phase'], 0, 0, 0, ic['repetition'],
                    ic['set'], ic['segment'], ic['slice'], ic['average'], 0] = data.T

    with open(str(hdr_path), 'w') as f:
        f.write('# Dimensions\n' + ' '.join(map(str, out_shape)) + '\n')

    return output_dir


def save_ormir_mids(output_dir: Path, stem: str, velocities_ms: np.ndarray, venc_ms: float,
                    affine: np.ndarray, tr_ms: float, config: dict,
                    magnitude: np.ndarray = None, mask: np.ndarray = None):
    """Save velocity data directly in ORMIR-MIDS format (mr-quant folder).

    velocities_ms: (x, y, z, t, 3) array in m/s
    venc_ms:       VENC in m/s
    affine:        4x4 affine matrix for the NIfTI header
    tr_ms:         repetition time in ms, used to compute TriggerTime array
    config:        sidecar JSON dict; VelocityEncodingDirection is read from it if present
    magnitude:     (x, y, z, t) reference encoding magnitude (optional)
    mask:          (x, y, z) boolean mask (optional)
    """
    print("\n[3/3] Saving ORMIR-MIDS format...")

    bids_dir = output_dir / "mr-quant"
    bids_dir.mkdir(exist_ok=True)

    n_timepoints = velocities_ms.shape[3]
    trigger_times = list(np.arange(n_timepoints) * tr_ms)

    vel_enc_dirs = config.get("VelocityEncodingDirection", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    nib.save(
        nib.Nifti1Image(velocities_ms.astype(np.float32), affine),
        str(bids_dir / f"{stem}_vel.nii.gz")
    )

    sidecar = {
        "FourthDimension": "TriggerTime",
        "TriggerTime": trigger_times,
        "FifthDimension": "VelocityEncodingDirection",
        "VelocityEncodingDirection": vel_enc_dirs,
        "Venc": [venc_ms, venc_ms, venc_ms],
    }

    with open(bids_dir / f"{stem}_vel.json", 'w') as f:
        json.dump(sidecar, f, indent=2)

    print(f"  {stem}_vel.nii.gz {velocities_ms.shape}")
    print(f"  {stem}_vel.json")

    if magnitude is not None:
        nib.save(
            nib.Nifti1Image(magnitude.astype(np.float32), affine),
            str(bids_dir / f"{stem}_part-mag.nii.gz")
        )
        mag_sidecar = {
            "FourthDimension": "TriggerTime",
            "TriggerTime": trigger_times,
        }
        with open(bids_dir / f"{stem}_part-mag.json", 'w') as f:
            json.dump(mag_sidecar, f, indent=2)
        print(f"  {stem}_part-mag.nii.gz {magnitude.shape}")
        print(f"  {stem}_part-mag.json")

    if mask is not None:
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), affine),
            str(bids_dir / f"{stem}_mask.nii.gz")
        )
        print(f"  {stem}_mask.nii.gz {mask.shape}")


def check_gpu_availability():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected")
            return True
        else:
            print("No NVIDIA GPU detected")
            return False
    except FileNotFoundError:
        print("nvidia-smi not found. No GPU support detected.")
        return False


def process_4dflow(
    dataset: ismrmrd.Dataset,
    config: dict,
    output_dir: Path,
    affine: np.ndarray,
    stem: str,
    use_gpu: bool = True,
    compressed_coils: int = 0,
    joint_recon: bool = False,
):
    """Full 4D flow pipeline: k-space conversion, reconstruction, ORMIR-MIDS output.

    dataset:          open ismrmrd.Dataset (caller is responsible for opening it)
    config:           dict loaded from the sidecar JSON file
    output_dir:       directory where BART temp files and the mr-quant output folder are written
    affine:           4x4 affine matrix for NIfTI output (e.g. from extract_geometry)
    stem:             base name used for intermediate and output files (e.g. mrd_path.stem)
    use_gpu:          enable BART GPU acceleration
    compressed_coils: number of virtual coils for coil compression (0 = disabled)
    joint_recon:      use joint phase/velocity reconstruction instead of per-phase
    """
    venc_ms = float(config.get("venc", 1.5))
    venc_cms = venc_ms * 100
    tr_ms = float(config.get("TR", 0.007)) * 1000
    ecalib = config.get("fullysampled_center", [13, 10])
    if isinstance(ecalib, str):
        ecalib = json.loads(ecalib)
    ecalib_r = ":".join(map(str, ecalib)) if isinstance(ecalib, list) else str(ecalib)

    output_dir.mkdir(parents=True, exist_ok=True)

    convert_mrd_to_bart(dataset, output_dir, stem)

    results = reconstruct_and_process_all(
        str(output_dir), venc_cms=venc_cms, use_gpu=use_gpu, ecalib_r=ecalib_r,
        compressed_coils=compressed_coils, joint_recon=joint_recon)

    for file_base, data in results.items():
        save_ormir_mids(output_dir, Path(file_base).name, data['velocities'], venc_ms, affine, tr_ms, config,
                        magnitude=np.mean(data['magnitude'], axis=-1), mask=data['mask'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct and process 4D flow MRI data.")
    parser.add_argument("mrd", type=str, help="Path to the .mrd input file.")
    parser.add_argument("json_config", type=str, help="Path to the sidecar JSON config file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to same directory as the MRD file).")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration.")
    parser.add_argument("--cc", type=int, default=0,
                        help="Number of compressed coils for the processing. Default: no compression")
    parser.add_argument("--joint", action="store_true", help="Perform joint reconstruction of phases and velocities")
    parser.add_argument("--keep-bart", action="store_true", help="Keep bart files")
    args = parser.parse_args()

    mrd_path = Path(args.mrd).resolve()
    output_dir = Path(args.output).resolve() if args.output else mrd_path.parent

    with open(args.json_config) as _f:
        config = json.load(_f)

    gpu_available = check_gpu_availability()
    use_gpu = gpu_available and not args.no_gpu
    if args.no_gpu:
        print("GPU acceleration disabled by user")
    elif not gpu_available:
        print("GPU not available, falling back to CPU")

    dataset = ismrmrd.Dataset(str(mrd_path))
    affine = extract_geometry(dataset, mrd_path)
    process_4dflow(dataset, config, output_dir, affine, stem=mrd_path.stem,
                   use_gpu=use_gpu, compressed_coils=args.cc, joint_recon=args.joint)

    if not args.keep_bart:
        for p in output_dir.glob("*.cfl"):
            p.unlink()
            hdr = p.with_suffix('.hdr')
            if hdr.exists():
                hdr.unlink()
        print("Cleaned up BART intermediate files.")
