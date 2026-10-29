class CG_bead():
    def __init__(self, beadindex=0, beadtype=None, resname=None, atom_indices=None):
        self._beadindex = beadindex
        self._beadtype = beadtype
        self._resname = resname
        self._atom_indices = atom_indices

    @property
    def beadindex(self):
        return self._beadindex

    @beadindex.setter
    def beadindex(self, beadindex):
        self._beadindex = beadindex

    @property
    def beadtype(self):
        return self._beadtype

    @beadtype.setter
    def beadtype(self, beadtype):
        self._beadtype = beadtype

    @property
    def atom_indices(self):
        return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, atom_indices):
        self._atom_indices = atom_indices

    @property
    def resname(self):
        return self._resname

    @resname.setter
    def resname(self, beadindex):
        self._resname = beadindex

