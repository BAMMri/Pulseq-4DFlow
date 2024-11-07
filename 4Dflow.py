import copy
import numpy as np
import pypulseq as pp
import sys
import gropt
import math
from scipy.signal import medfilt
from scipy.integrate import simpson


class MRISequence:
    def __init__(self, TE, TR, fov, Nx, Ny, Nz, Nslices, venc, slice_thickness, alpha, bandwidth, tbw):
        # Initialize sequence parameters
        self.TE = TE
        self.TR = TR
        self.fov = fov
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nslices = Nslices
        self.venc = venc
        self.slice_thickness= slice_thickness
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

    def smooth_gradient(self, time, amplitude, direction, start_smooth=None, end_smooth=None, filter_size=3, plot=False,
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
        smoothed_amplitude = targeted_smoothing(time, amplitude, start_smooth, end_smooth, filter_size)

        smoothed_area = simpson(smoothed_amplitude, x=time)
        if smoothed_area != 0:
            area_ratio = original_area / smoothed_area
            adjusted_smoothed_amplitude = smoothed_amplitude * area_ratio
        else:
            adjusted_smoothed_amplitude = smoothed_amplitude

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

        # Save the new coordinates
        smoothed_time = time
        smoothed_amplitude = adjusted_smoothed_amplitude

        return smoothed_time, smoothed_amplitude

    def make_gradient(self, axis, moment_params, flow_g_start=0, additional_params=None):

        M0 = moment_params[0][5]
        M1 = moment_params[1][5]

        # Create a tuple key for the cache
        cache_key = (axis, M0, M1)


        # Check if the gradient is already in the cache
        try:
            return self.gradient_cache[cache_key]
        except KeyError:
            pass

        if axis in ['y'] and abs(M0)*(self.sys.gamma * 1e-3) < 15:
            moment_params[0][5] = moment_params[0][5]*100
            moment_params[1][5] = moment_params[1][5]*100


        # If not in cache, create the gradient
        params = {
            'mode': 'free',
            'dt': self.seq.grad_raster_time,
            'gmax': 50,
            'smax': 110.0,
            'moment_params': moment_params,
            'TE': 3
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

            # Create the gradient object
        g_vel = pp.make_arbitrary_grad(channel=axis,
                                       waveform=G_waveform * self.sys.gamma,
                                       system=self.sys)

        # Store the gradient in the cache
        self.gradient_cache[cache_key] = copy.deepcopy(g_vel)

        return self.gradient_cache[cache_key]

    def make_tr(self, areay, areaz, m1x, m1y, m1z, labels=None):
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
        #Flow gy
        M0y = areay / self.sys.gamma * 1e6

        y_moment_params = [
            [0, 0, flow_g_start, -1, -1, M0y, 1.0e-4],
            [0, 1, flow_g_start, -1, -1, m1y, 1.0e-4]
        ]

        gy_vel = self.make_gradient('y', y_moment_params, flow_g_start=flow_g_start, additional_params={'gmax': 50, 'smax': 110.0})


        # Flow gx
        [M0x, M1x] = self.halftrap_m0_1(gx.amplitude / self.sys.gamma*1e3, gx.flat_time*1000 / 2, gx.rise_time*1000,
                                        (self.TE - pp.calc_duration(gx) / 2)*1000)

        x_moment_params = [
            [0, 0, flow_g_start, -1, -1, -M0x, 1.0e-4],
            [0, 1, flow_g_start, -1, -1, (m1x - M1x), 1.0e-4]
        ]

        gx_vel = self.make_gradient('x', x_moment_params, flow_g_start=flow_g_start,
                                         additional_params={'gmax': 50, 'smax': 110.0})

        #Flow gz
        [M0z, M1z] = self.halftrap_m0_1(gz.amplitude / self.sys.gamma*1e3, gz.flat_time*1000 / 2, gz.rise_time*1000, 0.00,
                                   second_half=True)

        M1z_total = m1z - M1z
        M0z_part = (areaz) / (self.sys.gamma)*1e6
        M0z_total = M0z_part - M0z

        z_moment_params = [
            [0, 0, flow_g_start, -1, -1,  M0z_total, 1.0e-6],
            [0, 1, flow_g_start, -1, -1,  M1z_total, 1.0e-6]
        ]

        gz_vel = self.make_gradient('z', z_moment_params, flow_g_start=flow_g_start,
                                         additional_params={'gmax': 50, 'smax': 110.0})
        #spoiling
        gx_spoil = pp.make_trapezoid(channel='x', area=2 * self.Nx * self.delta_kx, system=self.sys)
        gz_spoil = pp.make_trapezoid(channel='z', area=4 / self.slice_thickness-areaz, system=self.sys)

        #calculate delays
        delay_TE = math.ceil((self.TE - pp.calc_duration(gx_vel) - gz.fall_time - gz.flat_time / 2
                              - pp.calc_duration(gx) / 2) / self.seq.grad_raster_time) * self.seq.grad_raster_time

        delay_TR = math.ceil((self.TR - pp.calc_duration(gx_vel) - pp.calc_duration(gz)
                              - pp.calc_duration(gx) - delay_TE) / self.seq.grad_raster_time) * self.seq.grad_raster_time

        assert np.all(delay_TE >= 0)
        assert np.all(delay_TR >= 0)
        dTE = pp.make_delay(delay_TE)
        dTR = pp.make_delay(delay_TR)

        adc.dwell = np.round(adc.dwell / self.seq.adc_raster_time) * self.seq.adc_raster_time

        #smooth waveforms
        gx_vel.tt, gx_vel.waveform = self.smooth_gradient(gx_vel.tt, gx_vel.waveform, 'x', start_smooth=0.0e-3,
                                                     end_smooth=2.99e-3,  # 5.055e-3,
                                                     filter_size=11, plot=False, threshold=1e4)

        gz_vel.tt, gz_vel.waveform = self.smooth_gradient(gz_vel.tt, gz_vel.waveform, 'z', start_smooth=0.0e-3,
                                                     end_smooth=2.99e-3,  # 5.055e-3,
                                                     filter_size=11, plot=False, threshold=1e4)

        gy_vel.tt, gy_vel.waveform = self.smooth_gradient(gy_vel.tt, gy_vel.waveform, 'y',
                                                     start_smooth=0.0e-3,
                                                     end_smooth=2.99e-3,  # 5.055e-3,
                                                     filter_size=11, plot=False, threshold=1e4)

        #assemble sequence

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
        spoil_block_contents = [dTR, gy_reph, gx_spoil, gz_spoil]
        self.seq.add_block(*spoil_block_contents)


if __name__ == "__main__":

    import numpy as np
    import pypulseq as pp
    import sys

    import gropt

    import math
    import matplotlib.pyplot as plt
    from scipy.signal import medfilt
    from scipy.integrate import simpson
    from tqdm import tqdm

    DO_TRIGGERING=True
    PLOT_KSPACE=False
    HEART_RATE = 70

    RESOLUTION = [1.8e-3, 1.8e-3, 5e-3]
    FOV = [180e-3, 120e-3, 100e-3]
    VENC = 1.0

    from flow_comp_gre_class_F import MRISequence
    seq = MRISequence(
        TE=4.5e-3,
        TR=8e-3,
        fov=FOV,
        Nx=int(np.ceil(FOV[0] / RESOLUTION[0])),
        Ny=int(np.ceil(FOV[1] / RESOLUTION[1])),
        Nz=int(np.ceil(FOV[2] / RESOLUTION[2])),
        Nslices=6,
        venc=VENC,
        slice_thickness= 100e-3,
        alpha = 10,
        bandwidth= 1000,
        tbw = 2
    )


    # Calculate areay and areaz
    areay = (-(np.arange(seq.Ny) - seq.Ny / 2) * seq.delta_ky).tolist()
    areaz = (-(np.arange(seq.Nz) - seq.Nz / 2) * seq.delta_kz).tolist()

    # M1 values in mT*ms^2/m
    venc_values = [
        (0, 0, 0),
        (0.5e9 / (seq.sys.gamma * seq.venc), 0, 0),
        (0, 0.5e9 / (seq.sys.gamma * seq.venc), 0),
        (0, 0, 0.5e9 / (seq.sys.gamma * seq.venc))
    ]


    trig = pp.make_trigger(channel="physio1")
    if DO_TRIGGERING:
        max_phases = int(math.floor(60.0/HEART_RATE/(4*seq.TR)))
    else:
        max_phases = 1
    print(max_phases)
    # Main sequence loop
    for islice in tqdm(range(len(areaz))):

        for iphase in range(len(areay)):
            labels = []
            labels.append(pp.make_label(type="SET", label="PAR", value=islice))
            labels.append(pp.make_label(type="SET", label="LIN", value=iphase))
            if DO_TRIGGERING:
                seq.seq.add_block(trig)
            for cardiac_phase in range(max_phases): # max_phases are calculated as floor(cardiac_period / 4*TR)
                if DO_TRIGGERING:
                    labels.append(pp.make_label(type="SET", label="PHS", value=cardiac_phase))
                for tr_index, (M1x, M1y, M1z) in enumerate(venc_values):
                    labels.append(pp.make_label(type="SET", label="SET", value=tr_index))


                    seq.make_tr(areay[iphase], areaz[islice], M1x, M1y, M1z, labels)
                    labels.pop()
                if DO_TRIGGERING:
                    labels.pop()




    final_sequence = seq.seq

    # write sequence
    print('Sequence ready')
    seq.seq.set_definition('FOV', seq.fov)
    seq.seq.set_definition('Name', 'gre3d')
    seq.seq.write(f'MM_4Dflow_HR{HEART_RATE}_PHS{max_phases}_FOV{int(FOV[0]*1000)}x{int(FOV[1]*1000)}x{int(FOV[2]*1000)}_Venc{int(VENC*100)}.seq')

    if PLOT_KSPACE:
        # Show K-space sequence
        k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.seq.calculate_kspace()

        plt.figure()
        plt.plot(k_traj[0], k_traj[1])
        plt.plot(k_traj_adc[0], k_traj_adc[1], '.')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()

