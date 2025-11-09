
import subprocess
from pathlib import Path
# from config import SimulationConfig

class PedSim:
    DEFAULTS = {
        ".vcf": "input.vcf",
        ".vcf.gz": "input.vcf.gz",
        "def": "pedigree.def",
        "intf": "interference.tsv",
        "map": "input.simmap"
    }

    def __init__(self,
                executable_path: str,
                output: Path,
                vcf_ext: str = None,
                vcf_file: str = None,
                def_file: str = None,
                intf_file: str = None,
                map_file: str = None,
                founder_file: str = None):


        self.output = output
        self.prefix = output.name
        self.path = output.parent

        self.executable = executable_path

        if vcf_file is None:
            vcf_file = self.path / self.DEFAULTS[vcf_ext]
            def_file = self.path / self.DEFAULTS["def"]
            map_file = self.path / self.DEFAULTS["map"]
            intf_file = self.path / self.DEFAULTS["intf"]
        
        self.flags = {
            "-i": vcf_file,
            "-d": def_file,
            "--intf": intf_file,
            "-m": map_file,
            "--set_founders": founder_file,
            "-o": output
        }

        
    def update_flag(self, flag, arg):
        assert flag in self.flags

        self.flags[flag] = arg

    def get_file(self, file_type):

        file_types = ["fam"]

        assert file_type in file_types

        suffix = {
            "fam": "-everyone.fam"
        }

        return self.path / f"{self.prefix}{suffix[file_type]}"

    def get_input(self, file_type):

        file_types = {"vcf": "-i"}

        assert file_type in file_types

        return self.flags[file_types[file_type]]


    # @classmethod
    # def from_config(cls, config: SimulationConfig):

    #     config.pedsim.interference_file

    def _get_command(self, dry_run: bool = False):

        cmd = " ".join([f"{flag} {arg}" for flag, arg in self.flags.items() if arg])

        if dry_run:
            cmd += " --dry_run"

        return f"{self.executable} {cmd}"

    def _launch_command(self, pedsim_cmd: str):

        result = subprocess.run(
            pedsim_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True)

    # Execute the pedsim simulation
    def execute(self, dry_run: bool = False):

        cmd = self._get_command(dry_run=dry_run)

        self._launch_command(cmd)

    # Execute dry run
    def dry_run(self):
        self.execute(dry_run=True)