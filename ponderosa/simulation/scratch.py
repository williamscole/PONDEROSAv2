import subprocess




# from config import SimulationConfig

class PedSim:

    def __init__(self,
                 vcf_file: str,
                 def_file: str):
        
        self.flags = {
            "-i": vcf_file,
            "-d": def_file
        }

        self.executable = "blah"

    # @classmethod
    # def from_config(cls, config: SimulationConfig):

    #     config.pedsim.interference_file

    def _get_command(self, dry_run: bool = False):

        cmd = " ".join([f"{flag} {arg}" for flag, arg in self.flags.items()])

        if dry_run:
            cmd += " --dry_run"

        return f"{self.executable} {cmd}"

    def _launch_command(self, pedsim_cmd: str):

        result = subprocess.run(
            pedsim_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

    # Execute the pedsim simulation
    def execute(self, dry_run: bool = False):
        pass

    # Execute dry run
    def dry_run(self):
        
        self.execute(dry_run=True)