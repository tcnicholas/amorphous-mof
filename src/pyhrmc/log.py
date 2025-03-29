import gc
import pathlib
from datetime import datetime


class Log:
    def __init__(self, output_directory, name, update_log) -> None:
        """ """
        self.startTime = datetime.now()
        self.out = pathlib.Path(output_directory)
        self.out.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.update_log = int(update_log)
        self.struct_outdir = self.out / "struct"
        self.scattering_data = self.out / "scattering-data"
        self._make_log()
        self._make_folders()

    def _make_folders(self):
        """
        Try and keep the simulation output tidy.
        """
        # for structures, we reguarly deposit CIF and LAMMPS data.
        self.struct_outdir.mkdir(parents=True, exist_ok=True)
        (self.struct_outdir / "cif").mkdir(parents=True, exist_ok=True)
        (self.struct_outdir / "lammps-data").mkdir(parents=True, exist_ok=True)
        self.scattering_data.mkdir(parents=True, exist_ok=True)

    @property
    def log_header(self):
        """
        Create header for simulation details file.
        """
        h = (
            f"-----------------\n HRMC Simulation\n-----------------\n\n#s "
            f"Global summary:\n"
            f"\trun started : {self.startTime}\n"
            f"\toutput directory : '{self.out}'\n"
            f"\trun name : '{self.name}'\n"
            f"\tcomposition : '{self.box.composition.formula}'\n"
            f"\tformula units : "
            f"{self.box.composition.reduced_formula_and_factor[1]}\n"
            f"\twrite to log : {self.update_log}\n"
            f"\tcores: {self.cores}\n"
            f"#e\n\n"
        )
        print(h)
        h += self.energy_block()
        return h

    def energy_block(self):
        """Energy details block."""
        b = f"#s energy summary:\n\ttemperature: {self._t} K\n#e\n\n"
        print(b)
        return b

    def rdf_block(self, calculator, rmin, rmax, numr, binWidth):
        """
        RDF calculator block template.
        """
        b = (
            f"#s "
            f"rdf summary:\n"
            f"\tcalculator : {calculator}\n"
            f"\tr min/max : {rmin:.2f} {rmax:.2f}\n"
            f"\tnum r : {numr}\n"
            f"\tbin width : {binWidth}\n"
            f"#e\n\n"
        )
        print(b)
        self._append_to_log(b)

    def scattering_block(self, type, qmin, qmax, numq, weight):
        """Scattering information block template.

        Args
        ----
        type: str
            Type of scattering computed (xray or neutron).
        """
        b = (
            f"#s "
            f"{type} summary:\n"
            f"\tQ min/max : {qmin} {qmax}\n"
            f"\tnum Q : {numq}\n"
            f"\tweight : {weight}\n"
            f"#e\n\n"
        )
        print(b)
        self._append_to_log(b)

    def simulation_progress_header(self, costLabels):
        """ """
        h = f"{21 * '-'}\n Simulation progress\n{21 * '-'}\n"
        for l in [
            "Proposed",
            "TotalAccepted",
            "GoodMove",
            "BoltzmannMove",
            "Rejected",
        ]:
            h += f"{l:<20}"
        h += "| "
        for l in costLabels:
            h += f"{l:<20}"
        h += "| "
        h += "{:<20}".format("t (1M min/move)")
        h += "\n" + "-" * 207 + "\n"
        print(h)
        self._append_to_log(h)

    def write_simulation_progress(self):
        """
        Write simulation progress to log file.
        """

        # gather moves made information.
        movesDetails = [
            self.proposed,
            self.accepted,
            self.normal_accepted,
            self.boltzmann_accepted,
            self.rejected,
        ]

        # gather current costs.
        costs = []
        if self.neutron is not None:
            costs.append(self.neutron_cost(self.neutron.ones))
        if self.xray is not None:
            costs.append(self.xray_cost(self.xray.ones))
        if self._computeEnthalpy:
            costs.append(self.energyLast * self._temperature / self.box.natoms)

        # append total.
        costs.append(self.totalCost)

        # determine average time per 1000 moves.
        try:
            t = (
                (datetime.now() - self.startTime).total_seconds()
                * 1e6
                / self.proposed
                / 60
            )
        except:
            t = 0

        b = ""
        for v in list(movesDetails):
            b += f"{v:<20.0f}"
        b += "| "
        for v in list(costs):
            b += f"{v:<20.10}"
        b += "| "
        b += f"{t:<20.1f}"
        b += "\n"
        print(b)
        self._append_to_log(b)

        del costs, t, b
        gc.collect()

    def write_shuffle(self) -> None:
        """
        Write shuffle details.
        """
        b = f"#shuffle performed at {self.proposed} moves proposed.#"
        print(b)
        self._append_to_log(b)

    def finish(self):
        """Write final simulation details to log file."""

        # append to table.
        self._append_to_log("-" * 207 + "\n")
        self.write_simulation_progress()
        finishTime = datetime.now()

        b = (
            f"\n\n"
            f"--------------------\n Simulation summary\n--------------------\n"
            f"run finished : {finishTime}\n"
            f"elapsed time : {str(finishTime - self.startTime)}"
        )

        print(b)
        self._append_to_log(b)
        self.box.write_cif(self.out / "final_config.cif")
        self.box.write_data_file(self.out / "final_config_custom.data")
        self.lmp.command(
            f"write_data {str(self.out / 'final_config.data')} nocoeff"
        )
        self.lmp.close()

    def _make_log(self):
        """Create the log file to append to."""
        with open(self.out / "hrmc.dat", "w+") as f:
            f.write(self.log_header)

    def _append_to_log(self, block):
        """Append block to log file."""
        with open(self.out / "hrmc.dat", "a") as f:
            f.write(block)
