#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 02:08, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.hybrid.root_hybrid_ssnn import RootHybridSoniaSsnn
from mealpy.evolutionary_based import GA, DE, CRO
from mealpy.swarm_based import PSO, WOA, HHO
from mealpy.physics_based import MVO, TWO, EO
from mealpy.human_based import TLO, QSA
from mealpy.bio_based import IWO, SMA
from mealpy.system_based import AEO
from mealpy.math_based import SCA
from mealpy.music_based import HS


#####=============================== Evolutionary-based ================================================

class GaSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.pc = mha_paras["pc"]
        self.pm = mha_paras["pm"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.pc}-{self.pm}"

    def training(self):
        self.optimizer = GA.BaseGA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size, self.pc, self.pm)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class DeSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.wf = mha_paras["wf"]
        self.cr = mha_paras["cr"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.wf}-{self.cr}"

    def training(self):
        self.optimizer = DE.BaseDE(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size, self.wf, self.cr)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class CroSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.po = mha_paras["po"]
        self.Fb = mha_paras["Fb"]
        self.Fa = mha_paras["Fa"]
        self.Fd = mha_paras["Fd"]
        self.Pd = mha_paras["Pd"]
        self.G = mha_paras["G"]
        self.GCR = mha_paras["GCR"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.po}-{self.Fb}-{self.Fa}-{self.Fd}-{self.Pd}-{self.G}-{self.GCR}"

    def training(self):
        self.optimizer = CRO.BaseCRO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size,
                                     self.po, self.Fb, self.Fa, self.Fd, self.Pd, self.G, self.GCR)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== Swarm-based ================================================

class PsoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.c1 = mha_paras["c1"]
        self.c2 = mha_paras["c2"]
        self.w_min = mha_paras["w_min"]
        self.w_max = mha_paras["w_max"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.c1}-{self.c2}-{self.w_min}-{self.w_max}"

    def training(self):
        self.optimizer = PSO.BasePSO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size,
                                 self.c1, self.c2, self.w_min, self.w_max)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class WoaSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = WOA.BaseWOA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class HhoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = HHO.BaseHHO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== Physics-based ================================================

class MvoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.wep_minmax = mha_paras["wep_minmax"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = MVO.BaseMVO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size, self.wep_minmax)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class TwoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = TWO.BaseTWO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class EoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = EO.BaseEO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class IeoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = EO.LevyEO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== Human-based ================================================

class TloSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = TLO.BaseTLO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class QsaSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = QSA.BaseQSA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== Bio-based ================================================

class IwoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.seeds = mha_paras["seeds"]
        self.exponent = mha_paras["exponent"]
        self.sigma = mha_paras["sigma"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.seeds}-{self.exponent}-{self.sigma}"

    def training(self):
        self.optimizer = IWO.BaseIWO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size, self.seeds, self.exponent, self.sigma)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


class SmaSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.z = mha_paras["z"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.z}"

    def training(self):
        self.optimizer = SMA.BaseSMA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size, self.z)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== System-based ================================================

class AeoSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = AEO.BaseAEO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== Math-based ================================================

class ScaSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = SCA.BaseSCA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()


#####=============================== Music-based ================================================

class HsSonia(RootHybridSoniaSsnn):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_paras=None, sonia_paras=None, mha_paras=None):
        super().__init__(root_base_paras, root_hybrid_ssnn_paras, sonia_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.n_new = mha_paras["n_new"]
        self.c_r = mha_paras["c_r"]
        self.pa_r = mha_paras["pa_r"]
        self.filename = f"{self.filename}-{self.epoch}-{self.pop_size}-{self.n_new}-{self.c_r}-{self.pa_r}"

    def training(self):
        self.optimizer = HS.BaseHS(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size, self.n_new, self.c_r, self.pa_r)
        self.solution, self.best_fit, self.loss_train = self.optimizer.train()






