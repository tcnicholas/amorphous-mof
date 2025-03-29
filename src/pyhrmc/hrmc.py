# system libraries.
import gc
import math
from typing import Dict, Optional

# common packages.
import numpy as np
from pylammpsmpi import LammpsLibrary

# custom scripts.
from .box import Box
from .log import Log
from .move import Move
from .utils import mask_distances, append_array_to_file
from .total_scattering import TotalScattering

Boltzmann = 8.617333262e-5  # eV K-1


class HRMC(TotalScattering, Log):
    def __init__(
        self,
        input_file: str,
        atom_types: Dict[int, str],
        output_directory: str,
        dataFile: str,
        temperature: float = 298,  # Temperature in Kelvin.
        name: str = "HRMC_check",
        rMin: float = 0,  # Minimum radius, default is 0
        rMax: Optional[float] = None,
        binWidth: float = 0.02,
        update_log: int = 5000,
        force_rebuild: int = 1000,
        stagnant_check: int = 20000,
        cores: int = 1,
    ) -> None:
        # store a "random" number generator object.
        self.rnd = np.random.default_rng()

        # create LAMMPS instance.
        self.cores = cores
        self.lmp = LammpsLibrary(cores=cores, mode="local")
        self.lmp.file(input_file)

        # store simulation box information.
        self.box = Box(self.lmp, atom_types, name, dataFile)
        self._t = temperature

        # initialise total scattering class.
        TotalScattering.__init__(
            self, self.box.lengths, rMin=rMin, rMax=rMax, binWidth=binWidth
        )

        # initialise Log class.
        Log.__init__(self, output_directory, name=name, update_log=update_log)

        # self.lmp.command("compute 1 all pe")
        self._force_rebuild = force_rebuild

        # energy constant.
        self._computeEnthalpy = False
        self._temperature = temperature * Boltzmann
        self.e_constant = None
        self.hs_values = {}
        self.dist_tol = 0.0

        # store simulation progress.
        self.nbuilds = 0
        self.stagnant_check = stagnant_check
        self.stagnant = 0
        self.stagnant_last = 0

        self.proposed = 0
        self.accepted = 0
        self.normal_accepted = 0
        self.boltzmann_accepted = 0
        self.rejected = 0

        # store cost values.
        self.costs = None
        self.zerocost = None

        # ccost values.
        self.energyLast = 0.0
        self.xrayLast = 0.0
        self.neutronLast = 0.0
        self.totalCost = 0.0
        self.hs_violations = {}
        self.total_hs = 0.0

        # datafiles written.
        self._datafilenum = 1

    def setup_rdf_lammps(self):
        """
        Send the radial distribution function calculation fix to LAMMPS.

        This will make LAMMPS compute the radial distribution function, up to a
        given cut-off (defaults to half the size of the shortest box length) for
        all pairs of atom types.
        """

        # compute number of histogramming bins.
        nbins = int(self.rMax // self.binWidth) + 1

        # create compute rdf string for LAMMPS command. gets pairs of atoms
        rdf_str = ""
        for pair in self.box.element_pair_ix:
            rdf_str += " ".join(pair) + " "
        rdf_str = rdf_str.strip()

        # communicate with LAMMPS.
        self.lmp.command("neighbor 1.0 bin")
        self.lmp.command("neigh_modify delay 200 every 1 check yes")
        self.lmp.command("neigh_modify one 100000 page 1000000")
        self.lmp.command(f"comm_modify cutoff {self.rMax + 1}")
        self.lmp.command(
            f"compute rdf all rdf {nbins} {rdf_str} cutoff {self.rMax}"
        )
        self.lmp.command("run 1 pre yes post no")

        # extract initial rdf and store r-values.
        length = nbins
        width = len(self.box.element_pair_ix) * 2 + 1
        self.rdf_arr_size = (length, width)
        r = self.lmp.extract_compute("rdf", 0, 2, *self.rdf_arr_size)
        self.rs = r[:, 0]
        self.rdf = r[:, 1::2].T

        # append rdf block to log file.
        self.rdf_block(
            "lammps", self.rs[0], self.rs[-1], self.rs.shape[0], self.binWidth
        )

    def setup_hs(self, hs_values: dict, dist_tol=1e-3):
        """
        Setup calculations for hard-sphere constraints.
        """

        # store the HS values.
        self.hs_values = hs_values
        self.dist_tol = dist_tol

        # setup groups according to the index pairs in hs_values.
        for i, ix_pair in enumerate(hs_values.keys()):
            self.lmp.command(f"group {i} type {ix_pair[0]} {ix_pair[1]}")
            self.lmp.command(f"compute d{i} {i} pair/local dist")

        # run a calculation to check for current violations.
        self.lmp.command("run 1 pre yes post yes")
        for i, (ix_pair, rcut) in enumerate(hs_values.items()):
            nmax = (
                self.box.type_counts[ix_pair[0]]
                * self.box.type_counts[ix_pair[1]]
            )
            nmax *= 26
            ds = self.lmp.extract_compute(f"d{i}", 2, 1, nmax)
            viol = mask_distances(ds, nmax, dist_tol, rcut)
            self.hs_violations[ix_pair] = viol
        self.total_hs = np.sum(list(self.hs_violations.values()))

    def compute_hs(self):
        """
        Determine if move is hard-sphere acceptable.
        """
        hs_violations = {}
        for i, (ix_pair, rcut) in enumerate(self.hs_values.items()):
            nmax = (
                self.box.type_counts[ix_pair[0]]
                * self.box.type_counts[ix_pair[1]]
            )
            nmax *= 26
            ds = self.lmp.extract_compute(f"d{i}", 2, 1, nmax)
            viol = mask_distances(ds, nmax, self.dist_tol, rcut)
            hs_violations[ix_pair] = viol
        total_hs = np.sum(list(self.hs_violations.values()))

        if total_hs <= self.total_hs:
            accept = True
            self.hs_violations = hs_violations
            self.total_hs = total_hs
        else:
            accept = False
        return accept

    def compute_enthalpy(self):
        """Extract the current energy from LAMMPS."""
        return self.lmp.get_thermo("pe") * self.e_constant

    def compute_costs(self):
        if self.neutron is not None:
            neutron = self.neutron_cost(self.neutron.weight)
        else:
            neutron = 0.0
        if self.xray is not None:
            xray = self.xray_cost(self.xray.weight)
        else:
            xray = 0.0
        if self._computeEnthalpy:
            energy = self.compute_enthalpy()
        else:
            energy = 0

        return neutron, xray, energy, neutron + xray + energy

    def initialise(
        self,
        HD_ratio=1,
        computeEnergy=True,
    ):
        """
        Initialise calculations before beginning the simulation.
        """

        # initiate LAMMPS run.
        self.lmp.command("variable e equal pe")
        self.lmp.command("run 1 pre no post no")

        # store number of costs and labels (start at one to add a column for the
        # total cost).
        ncosts = 1
        costLabels = []

        # initialise neutron calculations.
        if self.neutron is not None:
            self.initialise_neutron(self.box, HD_ratio)
            ncosts += 1
            costLabels.append("S(Q)")

            if self.neutron.singleWeight:
                weight = self.neutron.weight[0]
            else:
                weight = "variable"

            self.scattering_block(
                "neutron",
                self.neutron.x.min(),
                self.neutron.x.max(),
                self.neutron.x.shape[0],
                weight,
            )

        # initialise xray calculations.
        if self.xray is not None:
            self.initialise_xray(self.box)
            ncosts += 1
            costLabels.append("F(Q)")

            if self.xray.singleWeight:
                weight = self.xray.weight[0]
            else:
                weight = "variable"

            self.scattering_block(
                "xray",
                self.xray.x.min(),
                self.xray.x.max(),
                self.xray.x.shape[0],
                weight,
            )

        # normalise the enthalpy by the kBT * natoms.
        if computeEnergy:
            self.computeEnthalpy = True
            self.e_constant = 1 / self._temperature
            ncosts += 1
            costLabels.append("Enthalpy (eV/atom)")

        # add total cost to label.
        costLabels.append("χ\u00b2")

        # alternatively, dont store costs in array, just store last value.
        if self.neutron is not None:
            self.neutronLast = self.neutron_cost(self.neutron.weight)
        if self.xray is not None:
            self.xrayLast = self.xray_cost(self.xray.weight)
        if self._computeEnthalpy:
            self.energyLast = self.compute_enthalpy()
        self.totalCost = self.neutronLast + self.xrayLast + self.energyLast

        # append to log file.
        self.simulation_progress_header(costLabels)
        self.write_simulation_progress()

    def run_minimisation(
        self, style="cg", etol=0.0, ftol=1.0e-8, maxiter=1000, maxeval=100000
    ):
        """
        Run a LAMMPS energy minimisation.
        """
        print(">>> running energy minimisation.")
        self.lmp.command(f"min_style {style}")
        self.lmp.command(f"minimize {etol} {ftol} {maxiter} {maxeval}")

    def force_moves(self, nmoves: int, maxTrans, maxRot, rigidBody=True):
        """
        Run random moves as you would during an ordinary refinement, except we
        accept all moves. This allows for creating a potentially more disordered
        structure (e.g. if starting from crystalline polymorph).
        """

        for _ in range(int(nmoves)):
            # choose random atom(s) and perturb them.
            if rigidBody:
                a_ids = self.rnd.choice(self.box.molecules)
            else:
                a_ids = [self.rnd.integers(1, self.box.natoms + 1)]

            # make the move.
            m = Move(self.box, self.lmp, a_ids)
            m.propose(maxTrans, maxRot)

    @property
    def temperature(self) -> float:
        """
        The current Monte Carlo temperature.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        """
        Set the temperature (K) of the simulation.

        This will be used when normalising the Boltzmann energy of the system
        """
        self._temperature = T * Boltzmann

    @property
    def computeEnthalpy(self):
        return self._computeEnthalpy

    @computeEnthalpy.setter
    def computeEnthalpy(self, val: bool):
        self._computeEnthalpy = val

    @property
    def force_rebuild(self):
        return self._force_rebuild

    @force_rebuild.setter
    def force_rebuild(self, val: int):
        self._force_rebuild = val

    def move(self, maxTrans, maxRot, rigidBody=True, hardSphere=False):
        """
        Perform RMC move.
        """

        self.proposed += 1

        # choose atoms to move. if rigid body is True, will select all atoms in
        # the given molecule and translate/rotate as one unit. otherwise, single
        # atoms from molecules can be moved (use with care if proper potentials
        # are not implemented).
        if rigidBody:
            a_ids = self.rnd.choice(self.box.molecules)
        else:
            a_ids = [self.rnd.integers(1, self.box.natoms + 1)]

        # make a move.
        m = Move(self.box, self.lmp, a_ids)
        m.propose(maxTrans, maxRot)

        # do we need to rebuild the neighbourlist?
        if (self.proposed % self._force_rebuild) == 0:
            self.lmp.command("run 0 pre yes post no")
        else:
            self.lmp.command("run 0 pre no post no")

        # check hard-shpere.
        accept_hs = True
        if hardSphere and self.hs_values.keys():
            accept_hs = self.compute_hs()
            if not accept_hs:
                m.reject()
                self.rejected += 1

        if accept_hs:
            if self.neutron is not None or self.xray is not None:
                rdf = self.lmp.extract_compute("rdf", 0, 2, *self.rdf_arr_size)[
                    :, 1::2
                ].T
                if self.rdf_smoothing is not None:
                    rdf += self.rdf_smoothing
                self.compute_scattering(rdf)
            neutron, xray, energy, totalCost = self.compute_costs()

            if totalCost <= self.totalCost:
                self.accepted += 1
                self.normal_accepted += 1
                self.neutronLast = neutron
                self.xrayLast = xray
                self.energyLast = energy
                self.totalCost = totalCost

                if self.neutron is not None or self.xray is not None:
                    self.rdf = rdf

            elif self.rnd.random() <= math.exp(self.totalCost - totalCost):
                self.accepted += 1
                self.boltzmann_accepted += 1
                self.neutronLast = neutron
                self.xrayLast = xray
                self.energyLast = energy
                self.totalCost = totalCost

                if self.neutron is not None or self.xray is not None:
                    self.rdf = rdf

            else:
                m.reject()
                self.rejected += 1

        del m, a_ids

        if self.proposed % self.update_log == 0:
            # update log file.
            self.write_simulation_progress()

            # create file string that specifies naccept, nreject, rather than
            # an abstract number associated with the dump frequency and the
            # number of proposed moves.
            file_name = f"{self._datafilenum}_p{self.proposed}_a{self.accepted}"

            # write structure files.
            self.box.write_cif(self.struct_outdir / f"cif/{file_name}.cif")
            self.box.write_data_file(
                self.struct_outdir / f"lammps-data/{file_name}.data"
            )

            # write scattering to file.
            if self.neutron is not None:
                append_array_to_file(
                    self.sq,
                    self.scattering_data / "neutron_structure_factor.txt",
                )

            if self.xray is not None:
                append_array_to_file(
                    self.fq, self.scattering_data / "xray_structure_factor.txt"
                )

            self._datafilenum += 1
            gc.collect()
