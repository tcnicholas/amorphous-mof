"""
22.12.21
@tcnicholas
Module for proposing new RMC moves on atoms and molecules.
"""

from typing import List

import numpy as np

import vector
from utils import ids2cids

rnd = np.random.default_rng()
moveTypes = np.array(["translate", "rotate"])


class Move:
    """
    A class for proposing and managing atomic and molecular moves in simulation.

    Attributes:
    ----------
    _box : object
        Simulation box object containing box dimensions.
    _lmp : object
        LAMMPS simulation instance.
    _a_ids : List[int]
        List of atom IDs involved in the move.
    _cids : ctypes.Array
        ctypes array representation of atom IDs.
    _lencids : int
        Number of atoms involved in the move.
    _xc_o : np.ndarray
        Original Cartesian coordinates of the atoms.
    _img_o : np.ndarray
        Original image flags for periodic boundary conditions.
    _xf_o : np.ndarray
        Fractional coordinates of the atoms (if applicable).
    _xc_n : np.ndarray
        Proposed new Cartesian coordinates of the atoms.
    _img_n : np.ndarray
        Proposed new image flags for periodic boundary conditions.

    Methods:
    -------
    propose(maxTrans: float, maxRot: float)
        Propose a translation or rotation move.
    reject()
        Revert atoms to their original positions if the move is rejected.
    _translate(maxTrans: float)
        Apply a translation move to the atoms.
    _rotate(maxRot: float)
        Apply a rotation move to the atoms.
    _gather_position()
        Gather the current positions and images of the atoms from LAMMPS.
    _send_position()
        Send the proposed positions and images to LAMMPS.
    """

    def __init__(self, box, lmp, a_ids: List[int]):
        """
        Initialise the Move object with the simulation box, LAMMPS instance, and
        atom IDs.

        Parameters:
        ----------
        box : object
            Simulation box containing box dimensions.
        lmp : object
            LAMMPS simulation instance.
        a_ids : List[int]
            List of atom IDs to move.
        """
        self._box = box
        self._lmp = lmp

        # Atom IDs
        self._a_ids = a_ids
        self._cids = ids2cids(a_ids)
        self._lencids = len(a_ids)

        # Old positions and images
        self._xc_o = None
        self._img_o = None

        # New positions and images
        self._xc_n = None
        self._img_n = None

        # Gather initial positions and images
        self._gather_position()

    @property
    def xc_o(self) -> np.ndarray:
        """Get original Cartesian coordinates."""
        return self._xc_o

    @property
    def img_o(self) -> np.ndarray:
        """Get original image flags."""
        return self._img_o

    def propose(self, maxTrans: float, maxRot: float) -> None:
        """
        Propose a new move (translation or rotation) for the atoms.

        Parameters:
        ----------
        maxTrans : float
            Maximum translation distance.
        maxRot : float
            Maximum rotation angle in degrees.
        """
        move = "translate" if self._lencids == 1 else rnd.choice(moveTypes)

        if move == "translate":
            self._translate(maxTrans)
        else:
            self._rotate(maxRot)

        # Wrap coordinates to stay within the simulation box
        self._img_n = (self._xc_n // self._box.lengths).astype(np.int32)
        self._xc_n -= self._img_n * self._box.lengths

        # Send the proposed positions to LAMMPS
        self._send_position()

    def reject(self) -> None:
        """
        Revert the atoms to their original positions if the move is rejected.
        """
        self._lmp.scatter_atoms("x", self._xc_o, ids=self._a_ids)
        self._lmp.scatter_atoms("image", self._img_o, ids=self._a_ids)

    def _translate(self, maxTrans: float) -> None:
        """
        Apply a translation move to the atoms.

        Parameters:
        ----------
        maxTrans : float
            Maximum translation distance.
        """
        translation_vector = (
            vector.random_vector(rnd.random(), rnd.random())
            * maxTrans
            * rnd.random()
        )
        self._xc_n += translation_vector

    def _rotate(self, maxRot: float) -> None:
        """
        Apply a rotation move to the atoms.

        Parameters:
        ----------
        maxRot : float
            Maximum rotation angle in degrees.
        """
        centre = np.mean(self._xc_n, axis=0)
        self._xc_n -= centre
        rotation_axis = vector.random_vector(rnd.random(), rnd.random())
        theta = vector.deg2rad(rnd.random() * maxRot)
        self._xc_n = (
            np.array(
                [vector.rodrigues(x, rotation_axis, theta) for x in self._xc_n]
            )
            + centre
        )

    def _gather_position(self) -> None:
        """
        Gather the current positions and image flags of the atoms from LAMMPS.
        """
        self._xc_o = self._lmp.gather_atoms("x", ids=self._a_ids)
        self._img_o = self._lmp.gather_atoms("image", ids=self._a_ids)
        self._xc_n = (self._img_o * self._box.lengths) + self._xc_o

    def _send_position(self) -> None:
        """
        Send the proposed new positions and image flags to LAMMPS.
        """
        self._lmp.scatter_atoms("x", self._xc_n, ids=self._a_ids)
        self._lmp.scatter_atoms("image", self._img_n, ids=self._a_ids)
