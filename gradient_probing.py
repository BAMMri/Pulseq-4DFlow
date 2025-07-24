

import copy

import numpy as np
import pypulseq as pp
import sys

import gropt
import math
from scipy.signal import medfilt
from scipy.integrate import simpson


class MRISequence:
    def __init__(self, TE, TR, fov, Nx, Ny, Nz, Nslices, slice_thickness, alpha, bandwidth, tbw):
        # Initialize sequence parameters
        self.TE = TE
        self.TR = TR
        self.fov = fov
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nslices = Nslices
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

        # Initialize sequence object
        self.sys = pp.Opts(max_grad=50, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
                      rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)
        self.seq = pp.Sequence(system=self.sys)


        # Initialize RF phase variables
        self.rf_phase = 0
        self.rf_inc = 0


    def make_tr(self, areay, areaz, tr_index, labels=None):

        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.alpha * np.pi / 180,
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

        gx_pre = pp.make_trapezoid(channel='x',
                                system=self.sys,
                                area=-gx.area / 2,
                                duration=1e-3)


        gx_flyback = pp.make_trapezoid(channel='x', area=-gx.area/2, system=self.sys)

        # Phase encoding gradient
        gy=pp.make_trapezoid(channel='y',
                            area=areay,
                            duration=1e-3,
                            system=self.sys,
                            )

        gy_reph = pp.make_trapezoid(channel='y',
                                    area=-areay,
                                    duration=1e-3,
                                    system=self.sys)


        gz_reph = pp.make_trapezoid(channel='z',
                                    area=areaz-gz.area/2,  #area=--gz.area/2 - areaz
                                    duration=1e-3,
                                    system=self.sys)

        #spoiling
        gx_spoil = pp.make_trapezoid(channel='x', area=2 * self.Nx * self.delta_kx, system=self.sys)
        gz_spoil = pp.make_trapezoid(channel='z', area=4 / self.slice_thickness-areaz, system=self.sys)

        #gx,gy,gz
        gx_grad = pp.make_trapezoid(channel='x', area=(self.Nx * self.delta_kx)/40, system=self.sys)
        gy_grad = pp.make_trapezoid(channel='y', area=(self.Ny * self.delta_ky)/40, system=self.sys)
        gz_grad = pp.make_trapezoid(channel='z', area=(self.Nz * self.delta_kz)/40, system=self.sys)



        adc.dwell = np.round(adc.dwell / self.seq.adc_raster_time) * self.seq.adc_raster_time



        #assemble sequence

        rf_1.phase_offset = self.rf_phase / 180 * np.pi
        adc.phase_offset = self.rf_phase / 180 * np.pi
        self.rf_inc = np.mod(self.rf_inc + self.RF_SPOIL_INC, 360.0)
        self.rf_phase = np.mod(self.rf_phase + self.rf_inc, 360.0)

        # Add blocks to sequence
        self.seq.add_block(rf_1, gz)

        # first TE
        delay_TE = self.TE[0] - pp.calc_duration(gx_pre,gy,gz_reph) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(gx) / 2


        delay_TE = math.ceil(delay_TE / self.seq.grad_raster_time) * self.seq.grad_raster_time
        local_labels = [pp.make_label(type="SET", label="ECO", value=0)]
        if labels:
            local_labels.extend(labels)

        self.seq.add_block(gy, gz_reph, gx_pre)
        self.seq.add_block(pp.make_delay(delay_TE), *local_labels)
        # Add readout gradient and ADC for the first TE
        self.seq.add_block(gx, adc)
        self.seq.add_block(gx_flyback)

        # Add cycling gradient before the second TE
        cycling_gradients = [gx_grad, gy_grad, gz_grad]
        cycling_grad = cycling_gradients[tr_index]

        self.seq.add_block(cycling_grad)

        delay_TE = self.TE[1] - self.TE[0] - pp.calc_duration(gx) - pp.calc_duration(gx_flyback) * 2 - pp.calc_duration(cycling_grad)
        delay_TE = math.ceil(delay_TE / self.seq.grad_raster_time) * self.seq.grad_raster_time
        self.seq.add_block(pp.make_delay(delay_TE), pp.make_label(type="INC", label="ECO", value=1))

        total_sequence_time = pp.calc_duration(gz)/2 + self.TE[1] + pp.calc_duration(gx)/2 + pp.calc_duration(gx_spoil, gz_spoil, gy_reph)
        delay_TR = math.ceil((self.TR - total_sequence_time) / self.seq.grad_raster_time) * self.seq.grad_raster_time

        assert np.all(delay_TE >= 0)

        assert np.all(delay_TR >= 0)

        self.seq.add_block(gx_flyback)

        # Add readout gradient and ADC for the second TE
        self.seq.add_block(gx, adc)
        assert np.all(delay_TE >= 0)
        # Add spoiling gradients and TR delay at the end of TR
        spoil_block_contents = [gy_reph, gx_spoil, gz_spoil]

        self.seq.add_block(*spoil_block_contents)
        self.seq.add_block(pp.make_delay(delay_TR))


if __name__ == "__main__":

    import numpy as np
    import pypulseq as pp
    import sys

    import math
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    PLOT_KSPACE=False
    from multi_te_test import MRISequence
    seq = MRISequence(
        TE=[2.5e-3, 5.5e-3],
        TR=10e-3,
        fov=[220e-3, 220e-3, 220e-3],
        Nx=64,
        Ny=64,
        Nz=64,
        Nslices=6,
        slice_thickness=200e-3,
        alpha=10,
        bandwidth=1000,
        tbw=2
    )


    # Calculate areay and areaz
    areay = (-(np.arange(seq.Ny) - seq.Ny / 2) * seq.delta_ky).tolist()
    areaz = (-(np.arange(seq.Nz) - seq.Nz / 2) * seq.delta_kz).tolist()



    # Main sequence loop
    for tr_index in range(3):
        for islice in tqdm(range(len(areaz))):

            for iphase in range(len(areay)):
                labels = []
                labels.append(pp.make_label(type="SET", label="PAR", value=islice))
                labels.append(pp.make_label(type="SET", label="LIN", value=iphase))
                labels.append(pp.make_label(type="SET", label="REP", value=tr_index))
                seq.make_tr(areay[iphase], areaz[islice], tr_index, labels)

    # write sequence
    print('Sequence ready')
    seq.seq.set_definition('FOV', seq.fov)
    seq.seq.set_definition('Name', 'gre3d')
    seq.seq.write('MM_probes_xyz_largeFOV.seq')

    if PLOT_KSPACE:
        # Show K-space sequence
        k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.seq.calculate_kspace()

        plt.figure()
        plt.plot(k_traj[0], k_traj[1])
        plt.plot(k_traj_adc[0], k_traj_adc[1], '.')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()



        seq.seq.plot(plot_now=False,time_range=np.array([0, 4]) * seq.TR, time_disp="ms")
        plt.figure(1)
        plt.show()

        seq.seq.plot(plot_now=False,time_range=np.array([0, 4]) * seq.TR, time_disp="ms")
        plt.figure(2)
        plt.show()

















