import numpy as np
import pypulseq as pp
from scipy.spatial import cKDTree
import sys
import copy

import gropt
import math
from scipy.signal import medfilt
from scipy.integrate import simpson
import scipy.spatial as spatial

import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg', 'WxAgg', etc.
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.stats import qmc

def setup_sampling(Ny, Nz, fov, undersampling_factor=9, center_percent_y=20, center_percent_z=35, seed=32):
    np.random.seed(seed)

    # Create k-space mask
    k_space = np.zeros((Ny, Nz), dtype=np.uint8)

    # Calculate fully sampled center dimensions based on FOV percentage
    fully_sampled_center_y = int((center_percent_y/100) * Ny / 2)
    fully_sampled_center_z = int((center_percent_z/100) * Nz / 2)

    # Fully sample the rectangular center
    center_y = Ny // 2
    center_z = Nz // 2
    start_y = center_y - fully_sampled_center_y
    end_y = center_y + fully_sampled_center_y + (Ny % 2)
    start_z = center_z - fully_sampled_center_z
    end_z = center_z + fully_sampled_center_z + (Nz % 2)

    k_space[start_y:end_y, start_z:end_z] = 1

    # Rest of the code remains the same...
    total_points = Ny * Nz
    center_points = (2 * fully_sampled_center_y) * (2 * fully_sampled_center_z)
    points_to_sample = int((total_points - center_points) / undersampling_factor)

    edge_points = [(0, Nz // 2), (Ny - 1, Nz // 2)]  # Add points at ky extremes

    for y, z in edge_points:
        if not k_space[y, z]:  # If not already in fully sampled region
            k_space[y, z] = 1
            points_to_sample -= 1

    # Calculate radius for Poisson disc sampling
    radius = max(1, int(np.sqrt((total_points - center_points) / (np.pi * points_to_sample))))

    # Generate Poisson disc samples for the outer region
    y_range = np.arange(Ny)
    z_range = np.arange(Nz)
    available_points = [(y, z) for y in y_range for z in z_range
                        if not (abs(y - center_y) < fully_sampled_center_y and
                                abs(z - center_z) < fully_sampled_center_z)]

    sampled_points = []
    while len(sampled_points) < points_to_sample and available_points:
        y, z = available_points.pop(np.random.randint(len(available_points)))

        if all(((y - center_y) - (sy - center_y)) ** 2 +
               ((z - center_z) - (sz - center_z)) ** 2 >= radius ** 2
               for sy, sz in sampled_points):
            sampled_points.append((y, z))
            k_space[y, z] = 1

        available_points = [(py, pz) for py, pz in available_points
                            if ((py - center_y) - (y - center_y)) ** 2 +
                            ((pz - center_z) - (z - center_z)) ** 2 >= radius ** 2]

    # Get all sampled coordinates
    y_indices, z_indices = np.where(k_space == 1)

    # Keep ky, kz positive (raw indices)
    ky_coords = y_indices
    kz_coords = z_indices

    return list(zip(ky_coords, kz_coords))


def generate_phase_samples(Ny, Nz, fov, num_phases, undersampling_factor=9, center_percent_y=20, center_percent_z=35, base_seed=32):
    phase_samples = []
    min_samples = float('inf')

    # Generate samples for all phases
    for phase in range(num_phases):
        phase_seed = base_seed + phase
        #samples = setup_sampling(Ny, Nz, fov, undersampling_factor,  fully_sampled_center_y, fully_sampled_center_z, seed=phase_seed)
        samples = setup_sampling(Ny, Nz, fov, undersampling_factor, center_percent_y, center_percent_z,seed=phase_seed)
        phase_samples.append(samples)
        min_samples = min(min_samples, len(samples))

    # Trim all phases to have the same number of samples
    phase_samples = [samples[:min_samples] for samples in phase_samples]
    # Make min_samples even to maintain symmetry
    # if min_samples % 2 != 0:
    #     min_samples -= 1
    #
    # phase_samples = [samples[:min_samples] for samples in phase_samples]

    return phase_samples


def is_within_ellipse(y, z, y_max, z_max, ellipse_radius):
    y_norm = (y - y_max / 2) / (y_max / 2)
    z_norm = (z - z_max / 2) / (z_max / 2)
    epsilon = 1e-10
    return (y_norm ** 2 + z_norm ** 2) <= (ellipse_radius ** 2 + epsilon)


class MRISequence:
    def __init__(self, TE, TR, fov, Nx, Ny, Nz, Nslices, venc, slice_thickness, alpha, bandwidth, tbw, heart_rate):
        # Initialize sequence parameters
        self.TE = TE
        self.TR = TR
        self.fov = fov
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nslices = Nslices
        self.venc = venc
        self.slice_thickness = slice_thickness
        self.alpha = alpha  # Flip angle in degrees
        self.bw = bandwidth  # Bandwidth
        self.tbw = tbw

        # Calculate derived parameters
        self.delta_kx = 1 / fov[0]
        self.delta_ky = 1 / fov[1]
        self.delta_kz = 1 / fov[2]

        # Constants
        self.RF_SPOIL_INC = 117.0
        self.gradient_cache = {}

        # Initialize sequence object
        self.sys = pp.Opts(max_grad=50, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
                           rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)
        self.seq = pp.Sequence(system=self.sys)

        # Initialize RF phase variables
        self.rf_phase = 0
        self.rf_inc = 0

        # Poisson disc sampling
        self.heart_rate = heart_rate
        self.max_phases = int(math.floor(60.0/self.heart_rate/(8*self.TR)))
        #self.max_phases = 1
        testphase_samples = generate_phase_samples(self.Ny, self.Nz, self.fov, self.max_phases, base_seed=32)
        self.phase_samples = np.array(testphase_samples)
        #self.max_phases = 1
        self.elliptical_scanning=True
        #self.samples, self.tree = setup_sampling(self.Nx, self.Ny, self.Nz, self.fov)

        # Scale and shift samples to match k-space dimensions
        #self.samples = (self.samples - 0.5) * 2 * np.pi / np.array(self.fov)  rad/m
        #self.samples = (self.samples - 0.5) * 2 / np.array(self.fov)
        filtered_points = {}
        print(self.phase_samples.shape)
        ellipse_radius = 1
        if self.elliptical_scanning:
            filtered_points = []
            min_points = float('inf')

            # First pass: filter points and find the minimum number of points across phases
            for cardiac_phase in range(self.phase_samples.shape[0]):
                phase_points = []
                for sample_index in range(self.phase_samples.shape[1]):
                    ky = self.phase_samples[cardiac_phase, sample_index, 0]
                    kz = self.phase_samples[cardiac_phase, sample_index, 1]

                    if is_within_ellipse(ky, kz, self.Ny, self.Nz, ellipse_radius):
                        phase_points.append(self.phase_samples[cardiac_phase, sample_index])

                filtered_points.append(phase_points)
                min_points = min(min_points, len(phase_points))

            # Second pass: truncate all phases to the minimum number of points
            self.phase_samples = np.array([phase[:min_points] for phase in filtered_points])

        print(self.phase_samples.shape)
            # ellipse_radius = 1
        # if self.elliptical_scanning:
        #     filtered_points = []
        #     for cardiac_phase in range(self.phase_samples.shape[0]):
        #         filtered_phase = self.phase_samples[cardiac_phase][
        #             [is_within_ellipse(ky, kz, self.Ny, self.Nz, ellipse_radius)
        #              for ky, kz in self.phase_samples[cardiac_phase]]
        #         ]
        #         filtered_points.append(filtered_phase)
        #
        #     self.phase_samples = np.array(filtered_points, dtype=object)


    def halftrap_m0_1(self, A, w, r, t0, second_half=False):
        absA = np.abs(A)
        s = A / r
        M0 = A * (w + r / 2)
        M1 = A * (absA ** 3 + (3 * w + 6 * t0) * s ** 2 * w * absA + (6 * w + 3 * t0) * s * A ** 2) / (
                6 * s ** 2 * absA)

        if second_half:
            M1 = A * (absA ** 3 + (3 * w + 6 * t0) * s ** 2 * w * absA + (3 * w + 3 * t0) * s * A ** 2) / (
                    6 * s ** 2 * absA)

        return M0, M1

    def smooth_gradient(self, time, amplitude, direction, start_smooth=None, end_smooth=None, filter_size=3, plot=True,
                        threshold=None):
        def targeted_smoothing(time, amplitude, start_smooth, end_smooth, filter_size, threshold=20):
            filtered = medfilt(amplitude, filter_size)
            if threshold is not None:
                filtered = np.where(np.abs(filtered) < threshold, 0, filtered)
            start_idx = np.where(time >= start_smooth)[0][0]
            end_idx = np.where(time >= end_smooth)[0][0]

            filtered[:start_idx] = np.linspace(filtered[0], filtered[start_idx], start_idx)
            filtered[end_idx:] = np.linspace(filtered[end_idx], filtered[-1], len(filtered) - end_idx)

            return filtered

        if start_smooth is None:
            start_smooth = time[int(len(time) * 0.1)]
        if end_smooth is None:
            end_smooth = time[int(len(time) * 0.9)]

        original_area = simpson(amplitude, x=time)

        if threshold is not None:
            n = len(amplitude)
            first_10_percent = int(n * 0.015)
            last_10_percent = n - first_10_percent

            # Apply threshold to first 10%
            amplitude[:first_10_percent] = np.where(
                np.abs(amplitude[:first_10_percent]) < threshold,
                0,
                amplitude[:first_10_percent]
            )

            amplitude[last_10_percent:] = np.where(
                np.abs(amplitude[last_10_percent:]) < threshold,
                0,
                amplitude[last_10_percent:]
            )
        # # if threshold is not None:
        # #     amplitude = np.where(np.abs(amplitude) < threshold, 0, amplitude)

        smoothed_amplitude = targeted_smoothing(time, amplitude, start_smooth, end_smooth, filter_size)

        smoothed_area = simpson(smoothed_amplitude, x=time)
        if smoothed_area != 0:
            area_ratio = original_area / smoothed_area
            adjusted_smoothed_amplitude = smoothed_amplitude * area_ratio
        else:
            adjusted_smoothed_amplitude = smoothed_amplitude
        # area_ratio = original_area / smoothed_area
        # adjusted_smoothed_amplitude = smoothed_amplitude * area_ratio

        # Plotting if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(time, amplitude, 'b-', label='Original')
            plt.subplot(1, 2, 2)
            plt.plot(time, adjusted_smoothed_amplitude, 'r-', label='Smoothed')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'MRI Trapezoid Gradient - {direction.upper()} Direction')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Save the new coordinates to NumPy arrays
        smoothed_time = time
        smoothed_amplitude = adjusted_smoothed_amplitude

        return smoothed_time, smoothed_amplitude

    def make_gradient(self, axis, moment_params, flow_g_start=0, additional_params=None):
        # Extract M0 and M1 from moment_params
        #M0 = moment_params[0][5] / (1e3 * moment_params[0][6])  # Reverse the scaling
        #M1 = moment_params[1][5] / (1e6 * moment_params[1][6])  # Reverse the scaling

        M0 = moment_params[0][5]
        M1 = moment_params[1][5]

    #     # Create a tuple key for the cache
        cache_key = (axis, M0, M1)


        # Check if the gradient is already in the cache
        # try:
        #     return self.gradient_cache[cache_key]
        # except KeyError:
        #     print(f"New Key: axis={axis}, M0={M0}, M1={M1}, flow_g_start={flow_g_start}")

        try:
            return self.gradient_cache[cache_key]
        except KeyError:
            #self.gradient_cache[cache_key] = cache_key
            #print(f"New Key: axis={axis}, M0={M0}, M1={M1}, flow_g_start={flow_g_start}")
            pass

        if axis in ['y'] and abs(M0)*(self.sys.gamma * 1e-3) < 15:
            moment_params[0][5] = moment_params[0][5]*100
            moment_params[1][5] = moment_params[1][5]*100


        # If not in cache, create the gradient
        params = {
            'mode': 'free',
            'dt': self.seq.grad_raster_time,
            'gmax': 50,
            'smax': 120.0,
            'moment_params': moment_params,
            'TE': 3.1
        }

        if additional_params:
            params.update(additional_params)

        G_waveform, _ = gropt.gropt(params)
        G_waveform = np.squeeze(G_waveform)

        # Add zero padding for y and z gradients
        if axis in ['y', 'z']:
            G_waveform = np.concatenate([np.array([0]), G_waveform, np.array([0])])

        if axis == 'y' and abs(M0)*(self.sys.gamma * 1e-3) < 15:
            G_waveform /= 100.0

        # Store the gradient in the cache
        #self.gradient_cache[cache_key] = G_waveform
        #print(f"Calculated gradient: {G_waveform}")
        #if axis == 'y':
            #G_waveform /= scale_factor

            # Create the gradient object
        g_vel = pp.make_arbitrary_grad(channel=axis,
                                       waveform=G_waveform * self.sys.gamma,
                                       system=self.sys)

        # Store the gradient in the cache
        self.gradient_cache[cache_key] = copy.deepcopy(g_vel)

        return self.gradient_cache[cache_key]

        #return G_waveform

    def make_tr(self, areay, areaz, m1x, m1y, m1z, labels=None):
        # areay = ky * (1/self.fov[1])
        # areaz = kz * (1/self.fov[2])

        # RF pulse
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=10 * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.slice_thickness,
            apodization=0.42,
            time_bw_product=self.tbw,
            system=self.sys,
            return_gz=True
        )



        # Readout gradient
        gx = pp.make_trapezoid(channel='x',
                               flat_area=self.Nx / self.fov[0],
                               flat_time=0.001,
                               system=self.sys)

        # ADC
        adc = pp.make_adc(num_samples=self.Nx,
                          delay=gx.rise_time,
                          duration=gx.flat_time,
                          system=self.sys)

        flow_g_start = pp.calc_duration(gz) / 2

        # Phase encoding gradient
        gy_reph = pp.make_trapezoid(channel='y',
                                    area=-areay,
                                    duration=1e-3,
                                    system=self.sys)
        # Flow gy
        M0y = areay / self.sys.gamma * 1e6
        # scale_factor = 100 if abs(areay) < 15 else 1
        scale_factor = 1
        y_moment_params = [
            [0, 0, flow_g_start, -1, -1, M0y * scale_factor, 1.0e-4],
            [0, 1, flow_g_start, -1, -1, m1y * scale_factor, 1.0e-4]
        ]

        gy_vel = self.make_gradient('y', y_moment_params, flow_g_start=flow_g_start,
                                    additional_params={'gmax': 50, 'smax': 120.0})

        # Flow gx
        [M0x, M1x] = self.halftrap_m0_1(gx.amplitude / self.sys.gamma * 1e3, gx.flat_time * 1000 / 2,
                                        gx.rise_time * 1000,
                                        (self.TE - pp.calc_duration(gx) / 2) * 1000)

        # print("M0x", M0x)

        # M1x is in mT*ms^2/m

        x_moment_params = [
            [0, 0, flow_g_start, -1, -1, -M0x, 1.0e-4],
            [0, 1, flow_g_start, -1, -1, (m1x - M1x), 1.0e-4]
        ]


        gx_vel = self.make_gradient('x', x_moment_params, flow_g_start=flow_g_start,
                                    additional_params={'gmax': 50, 'smax': 110.0})

        # Flow gz
        [M0z, M1z] = self.halftrap_m0_1(gz.amplitude / self.sys.gamma * 1e3, gz.flat_time * 1000 / 2,
                                        gz.rise_time * 1000, 0.00,
                                        second_half=True)

        def trap_m0_1(A, w, r, t0):
            absA = np.abs(A)
            s = A / r  # Slew rate

            # M0 calculation
            M0 = A * (w + r)  # Area of the full trapezoid

            # M1 calculation
            M1_ramp_up = (A * r ** 2) / 6  # Moment of the rising ramp
            M1_flat = (A * w * (2 * t0 + r + w)) / 2  # Moment of the flat top
            M1_ramp_down = (A * r * (3 * t0 + 3 * w + 2 * r)) / 6  # Moment of the falling ramp

            M1 = M1_ramp_up + M1_flat + M1_ramp_down
            return M0, M1

        M1z_total = m1z - M1z

        M0z_part = (areaz) / (self.sys.gamma) * 1e6
        # print("M0z_part", M0z_part)
        M0z_total = M0z_part - M0z
        z_moment_params = [
            [0, 0, flow_g_start, -1, -1, M0z_total, 1.0e-6],
            [0, 1, flow_g_start, -1, -1, M1z_total, 1.0e-6]
        ]
        # Gz_waveform = self.make_gradient('z', z_moment_params, flow_g_start=flow_g_start,
        #                                  additional_params={'gmax': 50, 'smax': 110.0})
        # gz_vel = pp.make_arbitrary_grad(channel='z',
        #                                 waveform=Gz_waveform * self.sys.gamma,
        #                                 system=self.sys)
        gz_vel = self.make_gradient('z', z_moment_params, flow_g_start=flow_g_start,
                                    additional_params={'gmax': 50, 'smax': 120.0})
        # spoiling
        gx_spoil = pp.make_trapezoid(channel='x', area=2 * self.Nx * self.delta_kx, system=self.sys)
        gz_spoil = pp.make_trapezoid(channel='z', area=4 / self.slice_thickness - areaz, system=self.sys)

        # calculate delays
        delay_TE = math.ceil((self.TE - pp.calc_duration(gx_vel) - gz.fall_time - gz.flat_time / 2
                              - pp.calc_duration(gx) / 2) / self.seq.grad_raster_time) * self.seq.grad_raster_time

        delay_TR = math.ceil((self.TR - pp.calc_duration(gx_vel) - pp.calc_duration(gz)
                              - pp.calc_duration(
                    gx) - delay_TE) / self.seq.grad_raster_time) * self.seq.grad_raster_time
        assert np.all(delay_TE >= 0)
        # delay_TR -= pp.calc_duration(gx_spoil, gz_spoil, gy_reph)
        assert np.all(delay_TR >= 0)

        # print("delay_TR", delay_TR)
        # print("spoiling_duration", pp.calc_duration(gx_spoil, gz_spoil))
        dTE = pp.make_delay(delay_TE)
        dTR = pp.make_delay(delay_TR)

        adc.dwell = np.round(adc.dwell / self.seq.adc_raster_time) * self.seq.adc_raster_time
        # smooth waveforms
        gx_vel.tt, gx_vel.waveform = self.smooth_gradient(gx_vel.tt, gx_vel.waveform, 'x', start_smooth=0.0e-3,
                                                          end_smooth=3.09e-3,  # 5.055e-3,
                                                          filter_size=11, plot=False, threshold=1e4)

        gz_vel.tt, gz_vel.waveform = self.smooth_gradient(gz_vel.tt, gz_vel.waveform, 'z', start_smooth=0.0e-3,
                                                          end_smooth=3.09e-3,  # 5.055e-3,
                                                          filter_size=11, plot=False, threshold=1e4)

        gy_vel.tt, gy_vel.waveform = self.smooth_gradient(gy_vel.tt, gy_vel.waveform, 'y',
                                                          start_smooth=0.0e-3,
                                                          end_smooth=3.09e-3,  # 5.055e-3,
                                                          filter_size=11, plot=False, threshold=1e4)
        # print(gy_vel.tt, gy_vel.waveform)
        # assemble sequence

        rf_1.phase_offset = self.rf_phase / 180 * np.pi
        adc.phase_offset = self.rf_phase / 180 * np.pi
        self.rf_inc = np.mod(self.rf_inc + self.RF_SPOIL_INC, 360.0)
        self.rf_phase = np.mod(self.rf_phase + self.rf_inc, 360.0)

        # Add blocks to sequence
        self.seq.add_block(rf_1, gz)
        self.seq.add_block(gy_vel, gx_vel, gz_vel)
        if labels:
            self.seq.add_block(dTE, *labels)
        else:
            self.seq.add_block(dTE)
        self.seq.add_block(gx, adc)

        # gy_reph[iphase].amplitude = -gy_reph[iphase].amplitude
        spoil_block_contents = [pp.make_delay(delay_TR), gy_reph, gx_spoil, gz_spoil]
        self.seq.add_block(*spoil_block_contents)
        # self.seq.add_block(dTR)




if __name__ == "__main__":

    import numpy as np
    import pypulseq as pp
    import sys

    import gropt

    import math
    import matplotlib.pyplot as plt
    from scipy.signal import medfilt
    from scipy.integrate import simpson
    import scipy.spatial as spatial
    from tqdm import tqdm

    DO_TRIGGERING=True
    PLOT_SEQUENCE=False
    PLOT_KSPACE=False
    TRIG_TIME = 80 #1.52

    RESOLUTION = [1.5e-3, 1.5e-3, 1.5e-3]
    FOV = [200e-3, 200e-3, 80e-3]
    VENC = 1.5

    from undersampling_arteries import MRISequence

    seq = MRISequence(TE=4.5e-3, TR=7e-3, fov=FOV, Nx=int(np.ceil(FOV[0] / RESOLUTION[0])),Ny=int(np.ceil(FOV[1] / RESOLUTION[1])),Nz=int(np.ceil(FOV[2] / RESOLUTION[2])),
                      Nslices=6, venc=VENC, slice_thickness=80e-3, alpha=10, bandwidth=1e3, tbw=2, heart_rate=TRIG_TIME)

    # M1 values in mT*ms^2/m
    venc_values = [
        (0, 0, 0),
        (0.5e9 / (seq.sys.gamma * seq.venc), 0, 0),
        (0, 0.5e9 / (seq.sys.gamma * seq.venc), 0),
        (0, 0, 0.5e9 / (seq.sys.gamma * seq.venc))
    ]

    trig = pp.make_trigger(channel="physio1")
    #trig = pp.make_trigger(channel="physio1",  duration=500e-6)
    if DO_TRIGGERING:
        max_phases = int(math.floor(60.0/TRIG_TIME/(8*seq.TR)))
        #max_phases = int(math.floor(TRIG_TIME / (8 * seq.TR)))
    else:
        max_phases = 1
    print(max_phases)
    print(seq.Nx, seq.Ny, seq.Nz)
    # Main sequence loop
    

    trigger_count = 0
    print(seq.phase_samples.shape)
    for sample_index in tqdm(range(0, seq.phase_samples.shape[1], 2)):
        if DO_TRIGGERING:
            seq.seq.add_block(trig)
            trigger_count += 1
        for cardiac_phase in range(seq.phase_samples.shape[0]):
            for line_offset in range(2):
                if sample_index + line_offset < seq.phase_samples.shape[1]:
                    ky = seq.phase_samples[cardiac_phase, sample_index + line_offset, 0]
                    kz = seq.phase_samples[cardiac_phase, sample_index + line_offset, 1]
                    # print("ky and kz=", ky, kz)
                    areay = (seq.Ny / 2 - ky) / seq.fov[1]
                    areaz = (seq.Nz / 2 - kz) / seq.fov[2]
                    print("areay and areaz=", areay, areaz)
                    labels = []
                    labels.append(pp.make_label(type="SET", label="PAR", value=int(kz)))
                    labels.append(pp.make_label(type="SET", label="LIN", value=int(ky)))
                    labels.append(pp.make_label(type="SET", label="PHS", value=cardiac_phase))

                    for tr_index, (M1x, M1y, M1z) in enumerate(venc_values):
                        labels.append(pp.make_label(type="SET", label="SET", value=tr_index))
                        seq.make_tr(areay, areaz, M1x, M1y, M1z, labels)
                        labels.pop()

        if DO_TRIGGERING:
            labels.pop()




    #print(seq.seq.test_report())
    # print('Sequence ready')
    # seq.seq.set_definition('FOV', seq.fov)
    # seq.seq.set_definition('Name', 'gre3d')
    # seq.seq.write(f'neurovasc_US9_vida_perc20_35_{TRIG_TIME}_PHS{max_phases}_FOV{int(FOV[0] * 1000)}x{int(FOV[1] * 1000)}x{int(FOV[2] * 1000)}_Venc{int(VENC * 100)}.seq')


    if PLOT_KSPACE:
        # Show K-space sequence
        k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.seq.calculate_kspace()
        np.save('k_traj_adc1.npy', k_traj_adc)

        plt.figure()
        #plt.plot(k_traj[1], k_traj[2])
        plt.plot(k_traj_adc[1], k_traj_adc[2], '.', alpha=0.5)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()


    if PLOT_SEQUENCE:

        seq.seq.plot(plot_now=False,time_range=np.array([0, 4]) * seq.TR, time_disp="ms")
        plt.figure(1)
        plt.show()

        seq.seq.plot(plot_now=False,time_range=np.array([0, 4]) * seq.TR, time_disp="ms")
        plt.figure(2)
        plt.show()




