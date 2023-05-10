import logging
import tempfile
import pexpect

import numpy as np
import pandas as pd

from pathlib import Path

from .utils import Timer


logger = logging.getLogger(__name__)


# foil = Foil(x, y)
# with foil.solve(reynolds=1e6, mach=0.5) as solver:
#   new_x, new_y = solver.repanel()
#   frame1 = solver.angles[0.5]
#   frame2 = solver.angles[0.1:0.9:0.25]


# 3. set-up quasi monte carlo, num configs as argument, count success rate
# 2. trailing edge finite, cosing spacing
# 4. visualise paneling before/after
# 1. raise for status
# 5. implement timing / num attempts log 
# 6. implement get item / slice syntactic sugar (options for loop over alfa or single aseq call)


class Foil:

    def __init__(self, x, y):
        # self.x = x
        # self.y = y
        self.coords = np.vstack([x, y]).T

    def __call__(self, mach=None, reynolds=5e4, normalize=True,
                 max_iter=200, max_retries=3, timeout=3):
        self.client = XFOILClient(timeout=timeout)

        self.mach = mach
        self.reynolds = reynolds

        self.max_retries = max_retries
        self.max_iter = max_iter
        self.timeout = timeout

        return self

    def __enter__(self):
        self.client.setup()
        self.client.disable_graphics()

        self.load()

        return self
        # self.angles = dude()

    def __exit__(self, type, value, traceback):
        self.client.teardown()

    def load(self, precision=8, header='foo'):
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as buffer:

            filename = buffer.name

            logger.info(f"Saving coordinates to temp file `{filename}`")
            np.savetxt(filename, self.coords, 
                       fmt=f'%.{precision:d}f', header=header, comments='')

            # TODO(ltiao): support filename or buffer as the Pandas IO utilities do
            self.client.load(filename)

    def calculate(self, key, raise_for_failure=False):

        assert isinstance(key, slice)

        self.client.enter_oper()

        if self.max_iter is not None:
            self.client.set_max_iter(self.max_iter)

        if self.reynolds is not None:
            self.client.enter_viscous_mode(self.reynolds)

        if self.mach is not None:
            self.client.set_mach(self.mach)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir).joinpath("result.dat")

            self.client.set_output(output_path)

            failures = []

            rows = []
            for alpha in np.mgrid[key]:

                logger.info(f"Solving for AoA {alpha:.3f}...")

                for i in range(self.max_retries):

                    with Timer() as timer:

                        self.client.child.sendline(f"ALFA {alpha:.3f}")

                        try:
                            self.client.child.expect("Point added to stored polar")
                            self.client.child.expect(fr"Point written to save file \s*{output_path}")
                        except pexpect.TIMEOUT:
                            logger.warning(f"Solver failed to converge in {self.timeout}s "
                                           f"on attempt {i+1}/{self.max_retries}")
                            continue  # this actually does nothing
                        else:
                            logger.info(f"Successfully solved for AoA {alpha:.3f} "
                                        f"on attempt {i+1}/{self.max_retries}")
                            break
                        finally:
                            logger.debug("Awaiting solver")
                            self.client.child.expect(r"\.OPER[iv]a \s*c>")
                            logger.debug("Solver ready to receive input!")

                else:
                    msg = (f"Failed to solve for AoA {alpha:.3f} "
                           f"after {self.max_retries} attempts!")
                    if raise_for_failure:
                        raise RuntimeError(msg)
                    logger.warning(msg)
                    failures.append(alpha)
                    continue

                row = dict(alpha=alpha, attempt=i, elapsed=timer.elapsed)
                rows.append(row)

            with output_path.open('r') as fh:
                data = pd.read_table(fh, skiprows=[*range(10), 11], 
                                     delim_whitespace=True, dtype="float64")  
                # .squeeze(axis='index')

            self.client.unset_output()

        metadata = pd.DataFrame(rows)

        if self.reynolds is not None:
            self.client.exit_viscous_mode()

        self.client.exit_oper()

        return data, metadata, failures

    def repanel(self, target_nodes=None, pane=True, ppar=False,
                pcop_on_failure=True):

        logger.info("Performing repaneling sequence...")

        if not (target_nodes is None or ppar):
            logger.warning(f"Target panel resolution {target_nodes:d} will "
                           f"be ignored as `ppar={ppar}`")

        success = True
        success &= self.client.pane(raise_for_warning=False, timeout=1.) if pane else True
        success &= self.client.ppar(target_nodes, raise_for_warning=False, timeout=1.) if ppar else True

        if not success:
            if pcop_on_failure:
                logger.info("Repaneling failed so restoring buffer airfoil")
                self.client.pcop()
            else:
                logger.info("Repaneling failed but not restoring buffer airfoil")

        # self.enter_gdes()
        # self.cadd()
        # self.exit_gdes()

        logger.info("Completed repaneling sequence")

        with tempfile.TemporaryDirectory() as temp_dir:
            # output_path = Path(buffer.name)
            output_path = Path(temp_dir).joinpath("after.dat")
            self.client.child.sendline(f"PSAV {output_path}")
            self.client.child.expect(r"XFOIL \s*c>")
            with output_path.open('rb') as infile:
                coords = np.loadtxt(infile)

        return coords


class XFOILClient:

    def __init__(self, timeout=5, graceful_terminate=True):
        self.graceful_terminate = graceful_terminate
        self.timeout = timeout

    def setup(self):
        # TODO(ltiao): allow user to specify alternative `xfoil` executable location
        logger.info("Spawning XFOIL session...")
        self.child = pexpect.spawn('xfoil', timeout=self.timeout)
        self.child.expect(r"XFOIL \s*c>")
        logger.info("Spawned XFOIL session")

    def teardown(self):
        logger.info("Terminating XFOIL session...")
        if self.graceful_terminate:
            logger.info("Attempting to terminate gracefully...")
            self.child.sendline("QUIT")
            self.child.expect(pexpect.EOF)
        self.child.close()
        logger.info("Terminated XFOIL session "
                    f"(exit status: {self.child.exitstatus}, "
                    f"signal status: {self.child.signalstatus})")

    def pcop(self):
        self.child.sendline('PCOP')
        self.child.expect(r"XFOIL \s*c>")

    def pane(self, raise_for_warning=False, timeout=None):
        self.child.sendline("PANE")
        try:
            self.child.expect("Paneling convergence failed.",
                              timeout=self.timeout if timeout is None else timeout)
        except pexpect.TIMEOUT:
            logger.info("Paneling resulted in no error messages.")
            return True
        else:
            logger.warn("Paneling convergence failed")
            if raise_for_warning:
                raise RuntimeError("Paneling failed to converge!")
            return False
        finally:
            self.child.expect(r"XFOIL \s*c>")

    def ppar(self, target_nodes=None, raise_for_warning=False, timeout=None):

        self.child.sendline("PPAR")
        self.child.expect(r"Change what \? \(<cr> if nothing else\) \s*c>")

        if target_nodes is None:
            self.child.sendline()
            self.child.expect(r"XFOIL \s*c>")
            return True
        else:
            logger.info(f"Setting target panels to {target_nodes:d}")

            self.child.sendline(f"N {target_nodes:d}")
            self.child.expect(r"Change what \? \(<cr> if nothing else\) \s*c>")

            self.child.sendline()
            try:
                self.child.expect("Paneling convergence failed.",
                                  timeout=self.timeout if timeout is None else timeout)
            except pexpect.TIMEOUT:
                logger.info("Paneling resulted in no error messages")
                return True
            else:
                logger.warn("Paneling convergence failed")
                if raise_for_warning:
                    raise RuntimeError("Paneling convergence failed")
                return False
            finally:
                self.child.expect(fr"N \s*i \s*Number of panel nodes \s*{target_nodes:d}")
                self.child.expect(r"Change what \? \(<cr> if nothing else\) \s*c>")
                self.child.sendline()
                self.child.expect(r"XFOIL \s*c>")

    def enter_gdes(self):
        self.child.sendline("GDES")
        self.child.expect(r"\.GDES \s*c>")

    def exit_gdes(self):
        self.child.sendline()
        # FIXME(ltiao): should not necessarily expect this
        self.child.expect("Buffer airfoil is not identical to current airfoil")
        self.child.expect(r"XFOIL \s*c>")

    def cadd(self):
        self.child.sendline("CADD")
        self.child.expect(r"Enter corner angle criterion for refinement \(deg\)")

        self.child.sendline()
        self.child.expect("Enter type of spline parameter")

        self.child.sendline()
        self.child.expect("Enter refinement x limits")

        self.child.sendline()
        self.child.expect(r"\.GDES \s*c>")

    def enter_oper(self):
        logger.info("Entering analysis mode")
        self.child.sendline("OPER")
        self.child.expect(r"\.OPERi \s*c>")

    def exit_oper(self):
        logger.info("Exiting analysis mode")
        # self.child.expect(r"\.OPERva \s*c>")
        self.child.sendline()
        self.child.expect(r"XFOIL \s*c>")

    def load(self, filename):
        logger.info(f"Loading coordinates from file `{filename}`"
                    " into XFOIL...")
        self.child.sendline(f"LOAD {filename}")
        failure = self.child.expect([
            "Current airfoil nodes set from buffer airfoil nodes",
            "Current airfoil cannot be set."
        ])
        if failure:
            raise RuntimeError("XFOIL failed to set airfoil nodes!")
        self.child.expect(r"XFOIL \s*c>")
        logger.info("Coordinates successfully loaded!")

    def unset_output(self):
        # self.child.expect(r"\.OPER[iv]a \s*c>")
        self.child.sendline("PACC")
        self.child.expect(r"\.OPER[iv] \s*c>")

    def set_output(self, output_path):
        self.child.sendline("PACC")
        self.child.expect(r"Enter \s*polar save filename  OR  <return> for no file \s*s>")

        logger.info(f"Solver results will be saved to `{output_path}`")

        self.child.sendline(str(output_path))

        self.child.expect("New polar save file available")
        self.child.expect(r"Enter \s*polar dump filename  OR  <return> for no file \s*s>")

        self.child.sendline()
        self.child.expect("Polar accumulation enabled")

        self.child.expect(r"\.OPER[iv]a \s*c>")

    def set_max_iter(self, max_iter):
        logger.info(f"Setting iteration limit {max_iter:d}")
        # child.expect("Enter new iteration limit \s*i>"); child.sendline("100")
        self.child.sendline(f"ITER {max_iter}")
        self.child.expect(r"\.OPERi \s*c>")

    def enter_viscous_mode(self, reynolds=None):
        logger.info(f"Entering viscous mode with Reynolds={reynolds:.3e}")
        # child.expect("Enter Reynolds number \s*r>"); child.sendline("1e6")
        self.child.sendline(f"VISC {reynolds:.3e}")
        self.child.expect("Re =")
        self.child.expect(r"\.OPERv \s*c>")

    def exit_viscous_mode(self):
        logger.info("Exiting viscous mode")
        # self.child.expect(r"\.OPERv \s*c>")
        self.child.sendline("VISC")
        self.child.expect(r"\.OPERi \s*c>")

    def set_mach(self, mach):
        logger.info(f"Setting Mach number={mach:.3f}")
        # child.expect("Enter Mach number \s*r>"); child.sendline(".6")
        self.child.sendline(f"MACH {mach:.3f}")
        self.child.expect(r"\.OPER[iv] \s*c>")

    def enable_normalization(self):
        self.child.sendline("NORM")
        self.child.expect(r"Loaded airfoil will \s*be normalized")
        self.child.expect(r"XFOIL \s*c>")

    def disable_graphics(self):
        logger.info("Disabling graphics")
        self.child.sendline("PLOP")
        self.child.expect("c>")

        self.child.sendline("G")
        self.child.expect(r"G raphics-enable flag:\s*F")
        self.child.expect("c>")

        self.child.sendline()
        self.child.expect(r"XFOIL \s*c>")
