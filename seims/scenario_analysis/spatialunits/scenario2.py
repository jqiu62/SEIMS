"""Scenario for optimizing BMPs based on slope position units.

    @author   : Liangjun Zhu, Huiran Gao

    @changelog:
    - 16-10-29  - hr - initial implementation.
    - 17-08-18  - lj - redesign and rewrite.
    - 18-02-09  - lj - compatible with Python3.
    - 23-06-29  - create gene list
"""
from __future__ import absolute_import, division, unicode_literals
from pickle import NONE
from future.utils import viewitems

import array
from collections import OrderedDict
from copy import deepcopy
import os
import sys
import random
import time
from struct import unpack
import json

from typing import Union, Dict, List, Tuple, Optional, Any, AnyStr
import numpy
from gridfs import GridFS
from pygeoc.raster import RasterUtilClass
from pygeoc.utils import FileClass, StringClass, UtilClass, get_config_parser, is_string
from pymongo.errors import NetworkTimeout
from pymongo import MongoClient

if os.path.abspath(os.path.join(sys.path[0], '../..')) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '../..')))

import global_mongoclient as MongoDBObj

from utility import read_simulation_from_txt, mask_rasterio
from preprocess.text import DBTableNames, RasterMetadata
from preprocess.sd_slopeposition_units import DelinateSlopePositionByThreshold
from scenario_analysis import _DEBUG, BMPS_CFG_UNITS, BMPS_CFG_METHODS
from scenario_analysis.scenario import Scenario
from scenario_analysis.config import SAConfig
from scenario_analysis.spatialunits.config import SASlpPosConfig, SAConnFieldConfig, \
    SACommUnitConfig


class SUScenario(Scenario):
    """Scenario analysis using different spatial units as BMPs configuration units."""

    def __init__(self, cf):
        # type: (Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig]) -> None
        """Initialization."""
        Scenario.__init__(self, cf)
        self.cfg = cf  # type: Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig]
        self.gene_num = cf.genes_num  # type: int
        self.gene_values = [0] * self.gene_num  # type: List[int, float] # 0 means no BMP

        # use gene list to store 1) bmp subscenario, 2) implementation year, 3) whether or not to maintain for the years after implementation
        if self.cfg.change_times > 0:
            self.change_times = self.cfg.change_times # make sure self.change_times is always positive
            self.gene_list = [[0, 0, [0] * self.change_times] for _ in range(self.gene_num)]
        self.base_amount = self.eval_info['BASE_ENV'] # for calculating environment-related, update in nsga2 base run

        self.bmps_params = dict()  # type: Dict[int, Any] # {bmp_subscenario id: {...}}
        self.suit_bmps = dict()  # type: Dict[AnyStr, Dict[int, List[int]]] # {type:{id: [bmp_ids]}}
        self.bmps_grade = dict()  # type: Dict[int, int] # {slppos_id: effectiveness_grade}

        self.read_bmp_parameters()
        bmps_suit_type = ['SLPPOS', 'LANDUSE'] \
            if self.cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3] else ['LANDUSE']
        self.get_suitable_bmps(bmps_suit_type)

    def setBaseEnvironment(self, number): # set base amount after running base scenario
        self.base_amount = number
        return self.base_amount

    def read_bmp_parameters(self):
        """Read BMP configuration from MongoDB.
        Each BMP is stored in Collection as one item identified by 'SUBSCENARIO' field,
        so the `self.bmps_params` is dict with BMP_ID ('SUBSCENARIO') as key.
        """
        conn = MongoDBObj.client  # type: MongoClient
        scenariodb = conn[self.scenario_db]

        bmpcoll = scenariodb[self.cfg.bmps_coll]
        # UserWarning: use an explicit session with no_cursor_timeout=True,
        # otherwise the cursor may still timeout after 30 minutes,
        # for more info see https://jira.mongodb.org/browse/DOCS-11255
        with conn.start_session() as session:
            for fb in bmpcoll.find(no_cursor_timeout=True, session=session):
                fb = UtilClass.decode_strs_in_dict(fb)
                if 'SUBSCENARIO' not in fb:
                    continue
                curid = fb['SUBSCENARIO']
                if curid not in self.cfg.bmps_subids:
                    continue
                if curid not in self.bmps_params:
                    self.bmps_params[curid] = dict()
                for k, v in fb.items():
                    if k == 'SUBSCENARIO':
                        continue
                    elif k == 'LANDUSE':
                        if isinstance(v, int):
                            v = [v]
                        elif v == 'ALL' or v == '':
                            v = None
                        else:
                            v = StringClass.extract_numeric_values_from_string(v)
                            v = [int(abs(nv)) for nv in v]
                        self.bmps_params[curid][k] = v[:]
                    elif k == 'SLPPOS':
                        if isinstance(v, int):
                            v = [v]
                        elif v == 'ALL' or v == '':
                            v = list(self.cfg.slppos_tags.keys())
                        else:
                            v = StringClass.extract_numeric_values_from_string(v)
                            v = [int(abs(nv)) for nv in v]
                        self.bmps_params[curid][k] = v[:]
                    elif k == 'INCOME':
                        if isinstance(v, int):  # scenario analysis
                            self.bmps_params[curid][k] = v
                        elif isinstance(v, str):  # bmp order optimization
                            v = StringClass.extract_numeric_values_from_string(v)
                            self.bmps_params[curid][k] = v[:]
                        else:
                            self.bmps_params[curid][k] = v
                    else:
                        self.bmps_params[curid][k] = v

    def get_suitable_bmps(self, types='LANDUSE'):
        # type: (Union[AnyStr, List[AnyStr]]) -> None
        """Construct the suitable BMPs for each slope position."""
        if is_string(types):
            types = [types]
        for bid, bdict in self.bmps_params.items():
            for type in types:
                if type not in bdict:
                    continue
                if type not in self.suit_bmps:
                    self.suit_bmps.setdefault(type, dict())
                suitsp = bdict[type]
                for sp in suitsp:
                    if sp not in self.suit_bmps[type]:
                        self.suit_bmps[type][sp] = [bid]
                    elif bid not in self.suit_bmps[type][sp]:
                        self.suit_bmps[type][sp].append(bid)
            if 'EFFECTIVENESS' in bdict:
                self.bmps_grade[bid] = bdict['EFFECTIVENESS']

    def initialize(self, input_genes=None):
        # type: (Optional[List]) -> List
        """Initialize a scenario.

        Returns:
            A list contains BMPs identifier of each gene location.
        """
        # Create configuration rate for each location randomly, 0.4 ~ 0.6
        cr = random.randint(40, 60) / 100.

        if input_genes is not None:  # Using the input genes
            if len(input_genes) == self.gene_num:
                self.gene_values = input_genes[:]
            else:  # Only usable for slope position units when optimizing unit boundary
                typenum = self.cfg.slppos_types_num
                tnum = self.cfg.thresh_num
                for idx, gv in enumerate(input_genes):
                    gidx = idx // typenum * (typenum + tnum) + idx % typenum
                    self.gene_values[gidx] = gv
            return self.gene_values
        else:
            if self.rule_mtd == BMPS_CFG_METHODS[0]:
                self.random_based_config(cr)
            else:
                self.rule_based_config(self.rule_mtd, cr)
        if self.cfg.boundary_adaptive and self.gene_num > self.cfg.units_num:
            # Randomly select boundary adaptive threshold
            thresholds = self.cfg.boundary_adaptive_threshs[:]
            for ti in range(self.gene_num):
                if ti in self.cfg.gene_to_unit:
                    continue
                if random.random() >= cr:
                    continue
                self.gene_values[ti] = thresholds[random.randint(0, len(thresholds) - 1)]
        if len(self.gene_values) == self.gene_num > 0:
            return self.gene_values
        else:
            raise RuntimeError('Initialize Scenario failed, please check the inherited scenario'
                               ' class, especially the overwritten rule_based_config and'
                               ' random_based_config!')

    # initialize gene list
    # once this function is called, gene_list and the coupled gene_values will be both generated
    def initialize_gene_list(self, input_gene_list = None):
        # type: (Optional[List]) -> List
        """Initialize a scenario as a gene list.
        Returns:
            A list contains BMPs identifier, implementation year, and maintenance plans of each gene location.
        """
        # Create configuration rate for each location randomly, 0.4 ~ 0.6
        cr = random.randint(40, 60) / 100.

        if input_gene_list is not None:  # Using the input genes
            if self.check_valid_gene_list(input_gene_list):
                self.gene_list = input_gene_list[:]
                self.generate_gene_values_from_gene_list()
            if all(len(sub) == 1 for sub in input_gene_list):
                self.initialize(input_gene_list) # if input as gene_values, first initialize gene_values then generate gene list
                self.generate_gene_list_from_gene_values()
            return self.gene_list

        else:
            if self.rule_mtd == BMPS_CFG_METHODS[0]:
                self.random_based_config_gene_list(cr)
            else:
                self.rule_based_config_gene_list(self.rule_mtd, cr)

        if len(self.gene_list) == len(self.gene_values) == self.gene_num > 0:
            return self.gene_list
        else:
            raise RuntimeError('Initialize Scenario as gene list failed, please check the inherited scenario'
                               ' class, especially the overwritten rule_based_config_gene_list and'
                               ' random_based_config_gene_list!')

    # check if the input gene list is valid for initializing
    def check_valid_gene_list(self, input_gene_list = None):
        valid = False
        if len(input_gene_list) == self.gene_num and all(len(sub) ==3 for sub in input_gene_list):
            for sub in input_gene_list:
                if sub[0] != 0:
                    valid = isinstance(sub[1], int) and 1 <= sub[1] <= self.change_times
        return valid

    # generate gene list from gene values, randomly assign implementation year and maintenance if bmp configured
    def generate_gene_list_from_gene_values(self):
        for i in range (self.gene_num):
           self.gene_list[i][0] = self.gene_values[i]
           # Must specify the implementation year if bmp has been configured. 
           # Randomly assign maintenance after the implementation year.
           if self.gene_list[i][0] != 0:
               self.gene_list[i][1] = random.randint(1, self.change_times)
               if self.gene_list[i][1] != self.change_times:
                   for ii in range(self.gene_list[i][1], self.change_times):
                       self.gene_list[i][2][ii] = random.randint(0, 1)

    # generate gene values from gene list, apply when taking input gene list directly from outside
    def generate_gene_values_from_gene_list(self):
        for i in range (self.gene_num):
           self.gene_values[i] = self.gene_list[i][0]

    def random_based_config(self, conf_rate=0.5):
        # type: (float) -> None
        """Config BMPs on each spatial unit randomly."""
        pot_bmps = self.cfg.bmps_subids[:]
        for uid, i in viewitems(self.cfg.unit_to_gene):
            if random.random() >= conf_rate:
                continue
            self.gene_values[i] = pot_bmps[random.randint(0, len(pot_bmps) - 1)]

    def random_based_config_gene_list(self, conf_rate = 0.5):
        # generate gene list randomly
        # first generate gene values then modify gene list
        self.random_based_config(conf_rate)
        self.generate_gene_list_from_gene_values()


    def rule_based_config(self, method, conf_rate=0.5):
        # type: (float, AnyStr) -> None
        """Config available BMPs on each spatial units by knowledge-based rule method.
        The looping methods vary from different spatial units, e.g., for slope position units,
        it is from the bottom slope position of each hillslope tracing upslope.

        The available rule methods are 'SUIT', 'UPDOWN', and 'HILLSLP'.

        See Also:
            :obj:`scenario_analysis.BMPS_CFG_METHODS`
        """
        if self.cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
            out_id = -1  # the last downstream unit ID
            for k, v in viewitems(self.cfg.units_infos['units']):
                if v['downslope'] <= 0:
                    out_id = k
                    break
            if out_id < 0:
                raise ValueError('The last downstream unit ID is not found!')
            cur_bmps = select_potential_bmps(out_id, self.suit_bmps['LANDUSE'],
                                             self.cfg.units_infos,
                                             self.cfg.unit_to_gene, self.gene_values,
                                             unit=self.cfg.bmps_cfg_unit,
                                             method=self.cfg.bmps_cfg_method)
            gene_idx = self.cfg.unit_to_gene[out_id]
            if cur_bmps is None or len(cur_bmps) == 0:
                self.gene_values[gene_idx] = 0
            else:
                self.gene_values[gene_idx] = cur_bmps[random.randint(0, len(cur_bmps) - 1)]
            if _DEBUG:
                print('-- Config BMPs for CONNFIELD+%s' % self.cfg.bmps_cfg_method)
                print('-- The most downstream ID is %d(gene index:%d)' % (out_id, gene_idx))
                print('-- Field ID, Gene Index, BMPs ID')
                print('-- %s, %s, %s' % (repr(out_id), repr(gene_idx),
                                         repr(self.gene_values[gene_idx])))
            up_units = self.cfg.units_infos['units'][out_id]['upslope'][:]
            unproceed = up_units[:]
            while len(unproceed) > 0:
                if _DEBUG:
                    print('-- unpreceed Field IDs: %s' % ','.join(repr(vv) for vv in unproceed))
                cur_unpreceed = list()
                for up_unit in unproceed:
                    gene_idx = self.cfg.unit_to_gene[up_unit]
                    cur_bmps = select_potential_bmps(up_unit, self.suit_bmps['LANDUSE'],
                                                     self.cfg.units_infos,
                                                     self.cfg.unit_to_gene, self.gene_values,
                                                     unit=self.cfg.bmps_cfg_unit,
                                                     method=self.cfg.bmps_cfg_method)
                    if cur_bmps is None or len(cur_bmps) == 0:
                        self.gene_values[gene_idx] = 0
                    elif random.random() > conf_rate:
                        self.gene_values[gene_idx] = 0
                    else:
                        self.gene_values[gene_idx] = cur_bmps[random.randint(0, len(cur_bmps) - 1)]
                    cur_upunits = self.cfg.units_infos['units'][up_unit]['upslope']
                    if _DEBUG:
                        print('-- %s, %s, %s' % (repr(up_unit), repr(gene_idx),
                                                 repr(self.gene_values[gene_idx])))
                        if -1 not in cur_upunits:
                            print('-- Upslope Field IDs of'
                                  ' %s is: %s' % (repr(up_unit),
                                                  ','.join(repr(vvv) for vvv in cur_upunits)))
                    for tmp_unit in cur_upunits:
                        if tmp_unit > 0:
                            cur_unpreceed.append(tmp_unit)
                unproceed = cur_unpreceed[:]
        elif self.cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
            spname = self.cfg.slppos_tagnames[-1][1]  # bottom slope position name, e.g., 'valley'
            for unitid, spdict in viewitems(self.cfg.units_infos[spname]):
                spidx = len(self.cfg.slppos_tagnames) - 1
                while True:  # trace upslope units
                    sptag = self.cfg.slppos_tagnames[spidx][0]
                    sp = self.cfg.slppos_tagnames[spidx][1]
                    up_spid = self.cfg.units_infos[sp][unitid]['upslope']
                    gene_idx = self.cfg.unit_to_gene[unitid]
                    spidx -= 1
                    # Get the union set of multiple suitable bmps
                    cur_suit_bmps = deepcopy(self.suit_bmps['SLPPOS'])
                    unit_area = self.cfg.units_infos[sp][unitid]['area']
                    unit_luids = self.cfg.units_infos[sp][unitid]['landuse']
                    lu_suit_bmps = self.suit_bmps['LANDUSE']
                    sp_suit_bmps = self.suit_bmps['SLPPOS'][sptag][:]
                    new_sp_suit_bmps = list()
                    for unit_luid, unit_luarea in viewitems(unit_luids):
                        if unit_luarea / unit_area < 0.1:
                            continue
                        if unit_luid not in lu_suit_bmps:
                            continue
                        for lu_suit_bmp in lu_suit_bmps[unit_luid]:
                            if lu_suit_bmp in sp_suit_bmps and lu_suit_bmp not in new_sp_suit_bmps:
                                new_sp_suit_bmps.append(lu_suit_bmp)
                    cur_suit_bmps[sptag] = new_sp_suit_bmps[:]

                    cur_bmps = select_potential_bmps(unitid, cur_suit_bmps, self.cfg.units_infos,
                                                     self.cfg.unit_to_gene, self.gene_values,
                                                     unit=self.cfg.bmps_cfg_unit,
                                                     method=self.cfg.bmps_cfg_method,
                                                     bmpgrades=self.bmps_grade,
                                                     tagnames=self.cfg.slppos_tagnames)
                    if cur_bmps is None or len(cur_bmps) == 0:
                        self.gene_values[gene_idx] = 0
                    elif random.random() > conf_rate:
                        # Do not config BMP according to probability
                        self.gene_values[gene_idx] = 0
                    else:
                        # config BMP
                        self.gene_values[gene_idx] = cur_bmps[random.randint(0, len(cur_bmps) - 1)]
                    if up_spid < 0:
                        break
                    unitid = up_spid
        else:
            # Loop each gene to config one of the suitable BMP
            for gene_idx in range(self.gene_num):
                unitid = self.cfg.gene_to_unit[gene_idx]
                cur_bmps = select_potential_bmps(unitid, self.suit_bmps['LANDUSE'],
                                                 self.cfg.units_infos,
                                                 self.cfg.unit_to_gene, self.gene_values,
                                                 unit=self.cfg.bmps_cfg_unit,
                                                 method=self.cfg.bmps_cfg_method,
                                                 bmpgrades=self.bmps_grade)
                if cur_bmps is None or len(cur_bmps) == 0:
                    self.gene_values[gene_idx] = 0
                    continue
                # select one randomly
                self.gene_values[gene_idx] = cur_bmps[random.randint(0, len(cur_bmps) - 1)]

    def rule_based_config_gene_list(self, method, conf_rate =0.5):
        self.rule_based_config(method, conf_rate)
        self.generate_gene_list_from_gene_values()

    def boundary_adjustment(self):
        """
        Update BMP configuration units and related data according to gene_values,
          i.e., bmps_info and units_infos
        """
        if not self.cfg.boundary_adaptive:
            return
        if self.gene_num == self.cfg.units_num:
            return
        # 1. New filename of BMP configuration unit
        dist = '%s_%d' % (self.cfg.orignal_dist, self.ID)
        self.bmps_info[self.cfg.bmpid]['DISTRIBUTION'] = dist
        spfilename = StringClass.split_string(dist, '|')[1]
        # 2. Organize the slope position IDs and thresholds by hillslope ID
        #    Format: {HillslopeID: {rdgID, bksID, vlyID, T_bks2rdg, T_bks2vly}, ...}
        slppos_threshs = dict()  # type: Dict[int, List]
        upperslppos = self.cfg.slppos_tagnames[0][1]  # Most upper slope position name
        for subbsnid, subbsndict in viewitems(self.cfg.units_infos['hierarchy_units']):
            for hillslpid, hillslpdict in viewitems(subbsndict):
                slppos_threshs[hillslpid] = list()
                for slppostag, slpposname in self.cfg.slppos_tagnames:
                    slppos_threshs[hillslpid].append(hillslpdict[slpposname])
                upper_geneidx = self.cfg.unit_to_gene[hillslpdict[upperslppos]]
                thresh_idx = upper_geneidx + len(hillslpdict)
                thresh_idxend = thresh_idx + self.cfg.thresh_num
                slppos_threshs[hillslpid] += self.gene_values[thresh_idx: thresh_idxend]
        # 3. Delineate slope position and get the updated information (landuse area, etc.)
        # 3.1 Erase current data in units_info
        for itag, iname in self.cfg.slppos_tagnames:
            if iname not in self.cfg.units_infos:
                continue
            for sid, datadict in viewitems(self.cfg.units_infos[iname]):
                self.cfg.units_infos[iname][sid]['area'] = 0.
                for luid in self.cfg.units_infos[iname][sid]['landuse']:
                    self.cfg.units_infos[iname][sid]['landuse'][luid] = 0.
        # 3.2 Delineate slope position and get data by subbasin
        # The whole watershed will be generateed for both version
        hillslp_data = DelinateSlopePositionByThreshold(self.modelcfg, slppos_threshs,
                                                        self.cfg.slppos_tag_gfs,
                                                        spfilename, subbsn_id=0)
        # 3.3 Update units_infos
        for tagname, slpposdict in viewitems(hillslp_data):
            for sid, datadict in viewitems(slpposdict):
                self.cfg.units_infos[tagname][sid]['area'] += hillslp_data[tagname][sid]['area']
                for luid in hillslp_data[tagname][sid]['landuse']:
                    if luid not in self.cfg.units_infos[tagname][sid]['landuse']:
                        self.cfg.units_infos[tagname][sid]['landuse'][luid] = 0.
                    newlanduse_area = hillslp_data[tagname][sid]['landuse'][luid]
                    self.cfg.units_infos[tagname][sid]['landuse'][luid] += newlanduse_area
        if self.modelcfg.version.upper() == 'MPI':
            for tmp_subbsnid in range(1, self.model.SubbasinCount + 1):
                DelinateSlopePositionByThreshold(self.modelcfg, slppos_threshs,
                                                 self.cfg.slppos_tag_gfs,
                                                 spfilename, subbsn_id=tmp_subbsnid)
        # print(self.cfg.units_infos)

    def decoding(self):
        """Decode gene values to Scenario item, i.e., `self.bmp_items`."""
        if self.ID < 0:
            self.set_unique_id()
        if self.bmp_items:
            self.bmp_items.clear()
        bmp_units = dict()  # type: Dict[int, List[int]] # {BMPs_ID: [units list]}
        for unit_id, gene_idx in viewitems(self.cfg.unit_to_gene):
            gene_v = self.gene_values[gene_idx]
            if gene_v == 0:
                continue
            if gene_v not in bmp_units:
                bmp_units[gene_v] = list()
            bmp_units[gene_v].append(unit_id)
        sce_item_count = 0
        for k, v in viewitems(bmp_units):
            curd = dict()
            curd['BMPID'] = self.cfg.bmpid
            curd['NAME'] = 'S%d' % self.ID
            curd['COLLECTION'] = self.bmps_info[self.cfg.bmpid]['COLLECTION']
            curd['DISTRIBUTION'] = self.bmps_info[self.cfg.bmpid]['DISTRIBUTION']
            curd['LOCATION'] = '-'.join(repr(uid) for uid in v)
            curd['SUBSCENARIO'] = k
            curd['ID'] = self.ID
            self.bmp_items[sce_item_count] = curd
            sce_item_count += 1
        # if BMPs_retain is not empty, append it.
        if len(self.bmps_retain) > 0:
            for k, v in viewitems(self.bmps_retain):
                curd = deepcopy(v)
                curd['BMPID'] = k
                curd['NAME'] = 'S%d' % self.ID
                curd['ID'] = self.ID
                self.bmp_items[sce_item_count] = curd
                sce_item_count += 1

    def decoding_from_gene_list(self):
        """Decode gene list to Scenario item, i.e., `self.bmp_items`."""
        if self.ID < 0:
            self.set_unique_id()
        if self.bmp_items:
            self.bmp_items.clear()
        bmp_units = dict()  # type: Dict[int, List[str]] # {subscenario ID: [unit|year|mt1:mt2, another]}
        for unit_id, gene_idx in viewitems(self.cfg.unit_to_gene):
            gene_bmp = self.gene_list[gene_idx][0]
            if gene_bmp == 0:
                continue
            # subscenario, year, maintenance
            subscenario, year, mt = self.gene_list[gene_idx]
            if subscenario not in bmp_units:
                bmp_units[subscenario] = list()
            bmp_units[subscenario].append('{0}|{1}|{2}'.format(unit_id, year, '|'.join(map(str, mt))))

        sce_item_count = 0
        for k, v in viewitems(bmp_units):
            curd = dict()
            curd['BMPID'] = self.cfg.bmpid
            curd['NAME'] = 'S%d' % self.ID
            curd['SUBSCENARIO'] = k
            curd['COLLECTION'] = self.bmps_info[self.cfg.bmpid]['COLLECTION']
            curd['DISTRIBUTION'] = self.bmps_info[self.cfg.bmpid]['DISTRIBUTION']
            curd['LOCATION'] = '-'.join(v)
            # curd['SUBSCENARIO'] = k
            curd['ID'] = self.ID
            curd['EFFECTIVENESSVARIABLE'] = 1 if self.cfg.effectiveness_changeable else 0
            curd['CHANGEFREQUENCY'] = self.cfg.change_frequency * 365 * 24 * 60 * 60  # convert to seconds
            curd['CHANGETIMES'] = self.cfg.change_times # append change times for C++ scenario
            curd['RAISE_BY_MT'] = self.bmps_params[k]['RAISE_BY_MT'] # store maintain raise values
            self.bmp_items[sce_item_count] = curd
            sce_item_count += 1
        # if BMPs_retain is not empty, append it.
        if len(self.bmps_retain) > 0:
            for k, v in viewitems(self.bmps_retain):
                curd = deepcopy(v)
                curd['BMPID'] = k
                curd['NAME'] = 'S%d' % self.ID
                curd['ID'] = self.ID
                curd['EFFECTIVENESSVARIABLE'] = 0
                curd['CHANGEFREQUENCY'] = -1
                curd['CHANGETIMES'] = 0
                self.bmp_items[sce_item_count] = curd
                sce_item_count += 1


    def import_from_mongodb(self, sid):
        pass

    def import_from_txt(self, sid):
        pass

    def export_scenario_to_gtiff(self, outpath=None):
        # type: (Optional[str]) -> None
        """Export scenario to GTiff.

        Read Raster from MongoDB should be extracted to pygeoc. -- Done using mask_rasterio!
        By ZhuLJ, 2023-03-25
        """
        if not self.export_sce_tif:
            return
        dist = self.bmps_info[self.cfg.bmpid]['DISTRIBUTION']
        dist_list = StringClass.split_string(dist, '|')
        if len(dist_list) >= 2 and dist_list[0] == 'RASTER':
            dist_name = '0_' + dist_list[1]  # prefix 0_ means the whole basin
            v_dict = dict()
            for unitidx, geneidx in viewitems(self.cfg.unit_to_gene):
                v_dict[unitidx] = self.gene_values[geneidx]
            if outpath is None:
                outpath = self.scenario_dir + os.path.sep + 'Scenario_%d.tif' % self.ID
            unit2bmpsstr = ','.join('%s:%s' % (repr(k), repr(v)) for k, v in v_dict.items())
            # print(unit2bmpsstr)
            mongoargs = [self.cfg.model.host, self.cfg.model.port,
                         self.cfg.model.db_name, 'SPATIAL']
            mask_rasterio(self.cfg.model.bin_dir,
                          [[dist_name, outpath, 0, -9999, 'INT32', unit2bmpsstr]],
                          mongoargs=mongoargs, maskfile='0_SUBBASIN', include_nodata=False)

    def calculate_economy(self):
        """Calculate economic benefit by simple cost-benefit model, see Qin et al. (2018)."""
        self.economy = 0.
        capex = 0.
        opex = 0.
        income = 0.
        actual_years = self.cfg.runtime_years
        for unit_id, gene_idx in viewitems(self.cfg.unit_to_gene):
            gene_v = self.gene_values[gene_idx]
            if gene_v == 0:
                continue
            unit_lu = dict()
            for spname, spunits in self.cfg.units_infos.items():
                if unit_id in spunits:
                    unit_lu = spunits[unit_id]['landuse']
                    break
            bmpparam = self.bmps_params[gene_v]
            for luid, luarea in unit_lu.items():
                if luid in bmpparam['LANDUSE'] or bmpparam['LANDUSE'] is None:
                    capex += luarea * bmpparam['CAPEX']
                    opex += luarea * bmpparam['OPEX'] * actual_years
                    income += luarea * bmpparam['INCOME'][-1] * actual_years

        # self.economy = capex
        # self.economy = capex + opex
        self.economy = capex + opex - income
        # print('economy: capex {}, income {}, opex {}'.format(capex, income, opex))
        return self.economy

    def calculate_economy_by_period(self, costs, maintains, incomes):
        """Calculate economic benefit by simple cost-benefit model, see Qin et al. (2018)."""
        self.net_costs_per_period = (costs + maintains - incomes).tolist()
        self.costs_per_period = (costs + maintains).tolist()
        self.incomes_per_period = incomes.tolist()
        
        self.economy = sum(self.net_costs_per_period)
        # print('economy:{}, capex {}, maintain {}, income {}'.format(self.economy, costs, maintains, incomes))
        return self.economy

    def calculate_environment(self):
        """Calculate environment benefit based on the output and base values predefined in
        configuration file.
        """
        if not self.modelrun:  # no evaluate done
            self.economy = self.worst_econ
            self.environment = self.worst_env
            return
        rfile = self.modelout_dir + os.path.sep + self.eval_info['ENVEVAL']

        if not FileClass.is_file_exists(rfile):
            time.sleep(0.1)  # Wait a moment in case of unpredictable file system error
        if not FileClass.is_file_exists(rfile):
            print('WARNING: Although SEIMS model has been executed, the desired output: %s'
                  ' cannot be found!' % rfile)
            self.economy = self.worst_econ
            self.environment = self.worst_env
            # model clean
            # self.model.SetMongoClient()
            # self.model.clean(delete_scenario=True)
            # self.model.UnsetMongoClient()
            return

        base_amount = self.base_amount
        if StringClass.string_match(rfile.split('.')[-1], 'tif'):  # Raster data
            rr = RasterUtilClass.read_raster(rfile)
            sed_sum = rr.get_sum() / self.eval_timerange  # unit: year
        elif StringClass.string_match(rfile.split('.')[-1], 'txt'):  # Time series data
            sed_sum = read_simulation_from_txt(self.modelout_dir,
                                               ['SED'], self.model.OutletID,
                                               self.cfg.eval_stime, self.cfg.eval_etime)
        else:
            raise ValueError('The file format of ENVEVAL MUST be tif or txt!')

        if base_amount < 0:  # indicates a base scenario
            self.environment = sed_sum
            self.sed_sum = sed_sum
        else:
            # reduction rate of soil erosion
            self.environment = (base_amount - sed_sum) / base_amount
            self.sed_sum = sed_sum
            # print exception values
            if self.environment > 1. or self.environment < 0. or self.environment is numpy.nan:
                print('Exception Information: Scenario ID: %d, '
                      'SUM(%s): %s' % (self.ID, rfile, repr(sed_sum)))
                self.environment = self.worst_env

    def calculate_environment_by_period(self):
        """Calculate environment benefit based on the output and base values predefined in
                configuration file.
                """
        # raster file names need reformat!
        # comment out sediment raster files by year, since error occurred. 06-29-2023
        if not self.modelrun:  # no evaluate done
            self.economy = self.worst_econ
            self.environment = self.worst_env
            return
        rfile = self.modelout_dir + os.path.sep + self.eval_info['ENVEVAL']

        if not FileClass.is_file_exists(rfile):
            time.sleep(0.1)  # Wait a moment in case of unpredictable file system error
        if not FileClass.is_file_exists(rfile):
            print('WARNING: Although SEIMS model has been executed, the desired output: %s'
                  ' cannot be found!' % rfile)
            self.economy = self.worst_econ
            self.environment = self.worst_env
            # model clean
            # self.model.SetMongoClient()
            # self.model.clean(delete_scenario=True)
            # self.model.UnsetMongoClient()
            return

        base_amount = self.base_amount
        sed_per_period = list()
        if StringClass.string_match(rfile.split('.')[-1], 'tif'):  # Raster data
            # sum of 2013-2017
            rr = RasterUtilClass.read_raster(rfile)
            sed_sum = rr.get_sum() / self.cfg.implementation_period  # Annual average of sediment 13-17
            for i in range(self.cfg.change_times):
                # 2013-2017
                filename = self.modelout_dir + os.path.sep + str(i+1) + '_' + self.eval_info['ENVEVAL']
                sed_per_period.append(RasterUtilClass.read_raster(filename).get_sum())
            # sed_sum = sed_per_period[-1]  # 2017 sed sum
        elif StringClass.string_match(rfile.split('.')[-1], 'txt'):  # Time series data
            sed_sum = read_simulation_from_txt(self.modelout_dir,
                                               ['SED'], self.model.OutletID,
                                               self.cfg.eval_stime, self.cfg.eval_etime)
        else:
            raise ValueError('The file format of ENVEVAL MUST be tif or txt!')

        if base_amount < 0:  # indicates a base scenario
            self.environment = sed_sum
            self.sed_sum = sed_sum
            self.sed_per_period = sed_per_period
        else:
            # reduction rate of soil erosion
            self.environment = (base_amount - sed_sum) / base_amount
            self.sed_sum = sed_sum
            self.sed_per_period = sed_per_period
            # print exception values
            if self.environment > 1. or self.environment < 0. or self.environment is numpy.nan:
                print('Exception Information: Scenario ID: %d, SUM(%s): %s, per period: %s'
                      % (self.ID, rfile, repr(sed_sum), sed_per_period))
                self.environment = self.worst_env
                    

    def calculate_profits_by_period(self):
        bmp_costs_by_period = numpy.array([0.] * self.change_times)
        bmp_maintain_by_period =  numpy.array([0.] * self.change_times)
        bmp_income_by_period =  numpy.array([0.] * self.change_times)
        for unit_id, gene_idx in viewitems(self.cfg.unit_to_gene):
            gene_v = self.gene_list[gene_idx][0]
            if gene_v == 0:
                continue
            unit_lu = dict()
            for spname, spunits in self.cfg.units_infos.items():
                if unit_id in spunits:
                    unit_lu = spunits[unit_id]['landuse']
                    break
            subscenario, year, mt = self.gene_list[gene_idx]
            bmpparam = self.bmps_params[subscenario]
            for luid, luarea in unit_lu.items():
                if luid in bmpparam['LANDUSE'] or bmpparam['LANDUSE'] is None:
                    opex = bmpparam['OPEX']
                    income = bmpparam['INCOME']
                    bmp_costs_by_period[year - 1] += luarea * bmpparam['CAPEX']
                    # every period has income after implementation
                    for prd in range(year - 1, self.cfg.change_times):  # closed interval
                        if mt[prd] == 1: # if maintenance has been adopted for the year, maintenance temporarily raises income
                            bmp_maintain_by_period[prd] += luarea * opex
                            bmp_income_by_period[prd] += luarea * income[prd - year + 1] * bmpparam['RAISE_BY_MT']
                        else:
                            bmp_income_by_period[prd] += luarea * income[prd - year + 1]  # each year has different benefit
        return bmp_costs_by_period, bmp_maintain_by_period, bmp_income_by_period

    def count_maintain_by_period(self):
        # count maintenance for each period
        mt_by_period = [0] * self.change_times
        for t in range(self.change_times):
            for gene in self.gene_list:
                if gene[2][t] == 1:
                    mt_by_period[t] += 1
        self.maintain_per_period = mt_by_period
        self.maintain_times = sum(mt_by_period)
        return mt_by_period

    def satisfy_investment_plan(self):
        # compute cost per period and compare with investment plans
        bmp_costs_by_period, bmp_maintain_by_period, bmp_income_by_period = self.calculate_profits_by_period()
        invest_limit = numpy.array(self.cfg.investment_each_period)
        costs = numpy.array(bmp_costs_by_period)
        maintain = numpy.array(bmp_maintain_by_period)
        income = numpy.array(bmp_income_by_period)
        remain = invest_limit - (costs + maintain - income)
        #print('investment limits: ', invest_limit)
        #print('costs: ', costs)
        #print('maintain: ', maintain)
        #print('income: ', income)
        #print('remain: ', remain)

        # not consider investment quota
        if not self.cfg.enable_investment_quota:
            print("Not considering investment limits.")
            return True, [costs, maintain, income]
        else:
            if self.cfg.investment_each_period is None:
                print("Investment plans are not provided!")
                return False, [None, None, None]

            # satisfy economic constraint
            if numpy.all(numpy.greater(invest_limit, costs + maintain - income)):
                self.net_costs_per_period = (costs + maintain - income).tolist()
                self.costs_per_period = (costs + maintain).tolist()
                self.incomes_per_period = income.tolist()
                return True, [costs, maintain, income]
            else:
                return False, [None, None, None]

    def adjust_year_for_invest(self):
        # try to adjust implementation years to satisfy investment limits
        satisfied, _ = self.satisfy_investment_plan()
        count = 0 # try in limited attempts
        while not satisfied: # be careful with dead loop
            invest = numpy.array(self.cfg.investment_each_period, dtype=float)
            prob_dist = invest / numpy.sum(invest)
            # do not change configuration type
            # assume the percent of implementation in each year correlates with investment distribution
            for idx, gene in enumerate(self.gene_list):
                rand_bit = numpy.random.choice(range(1, self.change_times + 1), p=prob_dist)
                gene[1] = rand_bit
                gene[2] = [0] * self.change_times
            satisfied, _ = self.satisfy_investment_plan()
            count += 1
            if count > 20:
                print("No configuration can satisfy the investment plan. Consider adjust the numbers.")
                break

    def statistics_by_period(self):
    # summary of each year, sum of each bmp for each year
        periods = list()
        for _ in range(self.change_times):
            bmps = dict()
            for bmpparam in self.bmps_params.values():
                temp_dict = dict()
                temp_dict['AREA'] = 0.
                temp_dict['CAPEX'] = 0.
                temp_dict['OPEX'] = 0.
                temp_dict['INCOME'] = 0.
                bmps[bmpparam['NAME']] = temp_dict
            periods.append({'SUMMARY': {}, 'BMPS': bmps})

        for unit_id, gene_idx in viewitems(self.cfg.unit_to_gene):
            gene_v = self.gene_list[gene_idx][0]
            if gene_v == 0:
                continue
            unit_lu = dict()
            for spname, spunits in self.cfg.units_infos.items():
                if unit_id in spunits:
                    unit_lu = spunits[unit_id]['landuse']
                    break
            subscenario, year, mt = self.gene_list[gene_idx]
            bmpparam = self.bmps_params[subscenario]
            for luid, luarea in unit_lu.items():
                if luid in bmpparam['LANDUSE'] or bmpparam['LANDUSE'] is None:
                    bmpname = bmpparam['NAME']
                    periods[year - 1]['BMPS'][bmpname]['CAPEX'] += luarea * bmpparam['CAPEX']
                    periods[year - 1]['BMPS'][bmpname]['AREA'] += luarea
                    for prd in range(year - 1, self.cfg.change_times):  # closed interval
                        implemented_index = prd - year + 1
                        if mt[prd] == 1:
                            periods[prd]['BMPS'][bmpname]['OPEX'] += luarea * bmpparam['OPEX']
                            periods[prd]['BMPS'][bmpname]['INCOME'] += luarea * bmpparam['INCOME'][implemented_index] * bmpparam['RAISE_BY_MT']
                        else:
                            periods[prd]['BMPS'][bmpname]['INCOME'] += luarea * bmpparam['INCOME'][implemented_index]

        for period in periods:
            total_capex = 0.
            total_opex = 0.
            total_income = 0.
            total_area = 0.
            for bmp_detail in period['BMPS'].values():
                total_capex += bmp_detail['CAPEX']
                total_opex += bmp_detail['OPEX']
                total_income += bmp_detail['INCOME']
                total_area += bmp_detail['AREA']
            period['SUMMARY']['CAPEX'] = total_capex
            period['SUMMARY']['OPEX'] = total_opex
            period['SUMMARY']['INCOME'] = total_income
            period['SUMMARY']['NETCOST'] = total_capex + total_opex - total_income
            period['SUMMARY']['AREA'] = total_area

        return periods

    def statistics_by_bmp(self):
        bmps = dict() #{name:{area:, income:}}
        for bmpparam in self.bmps_params.values():
            temp_dict = dict()
            temp_dict['AREA'] = 0.
            temp_dict['CAPEX'] = 0.
            temp_dict['OPEX'] = 0.
            temp_dict['INCOME'] = 0.
            bmps[bmpparam['NAME']] = temp_dict        

        for unit_id, gene_idx in viewitems(self.cfg.unit_to_gene):
            gene_v = self.gene_list[gene_idx][0]
            if gene_v == 0:
                continue
            unit_lu = dict()
            for spname, spunits in self.cfg.units_infos.items():
                if unit_id in spunits:
                    unit_lu = spunits[unit_id]['landuse']
                    break
            subscenario, year, mt = self.gene_list[gene_idx]
            bmpparam = self.bmps_params[subscenario]
            for luid, luarea in unit_lu.items():
                if luid in bmpparam['LANDUSE'] or bmpparam['LANDUSE'] is None:
                    bmpname = bmpparam['NAME']
                    bmps[bmpname]['CAPEX'] += luarea * bmpparam['CAPEX']
                    bmps[bmpname]['AREA'] += luarea
                    for prd in range(year - 1, self.cfg.change_times):  # closed interval
                        if mt[prd] == 1:
                            bmps[bmpname]['OPEX'] = luarea * bmpparam['OPEX']
                            bmps[bmpname]['INCOME'] += luarea * bmpparam['INCOME'][prd - year + 1] * bmpparam['RAISE_BY_MT']
                        else:
                            bmps[bmpname]['INCOME'] += luarea * bmpparam['INCOME'][prd - year + 1]
                        
        return bmps


def select_potential_bmps(unitid,  # type: int
                          suitbmps,  # type: Dict[int, List[int]] # key could be SLPPOS or LANDUSE
                          unitsinfo,  # type: Dict[Union[AnyStr, int], Any]
                          unit2gene,  # type: OrderedDict[int, int]
                          ind,  # type: Union[array.array, List[int], Tuple[int]] # gene values
                          unit='SLPPOS',  # type: AnyStr
                          method='SUIT',  # type: AnyStr
                          bmpgrades=None,  # type: Optional[Dict[int, int]]
                          tagnames=None  # type: Optional[List[Tuple[int, AnyStr]]] # for SLPPOS
                          ):
    # type: (...) -> Optional[List[int]]
    """Select potential BMPs for specific spatial unit."""
    suit_bmps_tag = -1
    down_unit = -1  # type: Optional[int]
    up_units = list()  # type: Optional[List[int]]
    if unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        for spid, spdict in viewitems(unitsinfo):
            if unitid not in spdict:
                continue
            down_unit = spdict[unitid].get('downslope')
            up_units.append(spdict[unitid].get('upslope'))
            for t, n in tagnames:
                if spid == n:
                    suit_bmps_tag = t
                    break
    else:  # other spatial units only take `LANDUSE` to suit BMPs
        # ValueError checks should be done in other place
        suit_bmps_tag = unitsinfo['units'][unitid]['primarylanduse']
        down_unit = unitsinfo['units'][unitid].get('downslope')  # may be None
        up_units = unitsinfo['units'][unitid].get('upslope')  # may be None

    if suit_bmps_tag not in suitbmps:
        return None

    bmps = suitbmps[suit_bmps_tag][:]
    bmps = list(set(bmps))  # ascending
    # Config or not is controlled by Random probability outside this function,
    #  thus, there is no need to append 0 (i.e., no BMP)!
    # if 0 not in bmps:
    #     bmps.append(0)

    if method == BMPS_CFG_METHODS[0] or method == BMPS_CFG_METHODS[1]:  # RDM or SUIT
        return bmps

    down_position = False
    down_gvalue = -1
    if down_unit is not None and down_unit > 0:
        down_gvalue = ind[unit2gene[down_unit]]
    else:
        down_position = True

    if method == BMPS_CFG_METHODS[2]:  # UPDOWN
        if down_unit <= 0:  # If downslope unit does not exists
            return bmps
        upslope_configured = False
        for upslope_id in up_units:
            if upslope_id < 0:
                continue
            if ind[unit2gene[upslope_id]] > 0:
                upslope_configured = True
        if down_gvalue > 0:
            if _DEBUG:
                print('  Mutate on unit: %d, the downslope unit has been configured BMP.' % unitid)
            # If downslope unit is configured BMP, then this unit should not configure BMP
            bmps = [0]
        elif upslope_configured:
            if _DEBUG:
                print('  Mutate on unit: %d, at least one of the upslope units '
                      'has been configured BMP.' % unitid)
            # If downslope unit is configured BMP, then this unit should not configure BMP
            bmps = [0]
        else:
            # If downslope unit is not configured BMP and the upslope units are all not configured
            #  BMPs, then this unit will be configured one BMP
            if 0 in bmps:
                bmps.remove(0)
        return bmps

    if method == BMPS_CFG_METHODS[3]:  # SLPPOS
        if bmpgrades is None:  # By default, the effectiveness grade should be equal for all BMPs.
            bmpgrades = {bid: 1 for bid in bmps}
        if 0 not in bmpgrades:
            bmpgrades[0] = 0

        top_position = False
        up_gvalue = -1
        if up_units is not None and up_units[0] > 0:
            up_gvalue = ind[unit2gene[up_units[0]]]
        else:
            top_position = True

        up_grade = bmpgrades[up_gvalue] if up_gvalue in bmpgrades else 0
        down_grade = bmpgrades[down_gvalue] if down_gvalue in bmpgrades else 0
        new_bmps = list()
        if top_position and down_gvalue > 0:  # 1. the top slppos, and downslope with BMP
            for _bid, _bgrade in viewitems(bmpgrades):
                if _bgrade <= down_grade and _bid in bmps:
                    new_bmps.append(_bid)
        elif down_position and up_gvalue > 0:  # 2. the bottom slppos, and upslope with BMP
            for _bid, _bgrade in viewitems(bmpgrades):
                if up_grade <= _bgrade and _bid in bmps:
                    new_bmps.append(_bid)
        elif not top_position and not down_position:  # 3. middle slppos
            for _bid, _bgrade in viewitems(bmpgrades):
                if down_gvalue == 0 and up_gvalue <= _bgrade and _bid in bmps:
                    # 3.1. downslope no BMP
                    new_bmps.append(_bid)
                elif up_grade <= _bgrade <= down_grade and _bid in bmps:
                    # 3.2. downslope with BMP
                    new_bmps.append(_bid)
        else:  # Do nothing
            pass
        if len(new_bmps) > 0:
            bmps = list(set(new_bmps))
        return bmps

    return bmps  # for other BMP configuration methods, return without modification


def initialize_scenario(cf, input_genes=None):
    # type: (Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig], Optional[List]) -> List[int]
    """Initialize gene values"""
    sce = SUScenario(cf)
    return sce.initialize(input_genes=input_genes)


def initialize_scenario_gene_list(cf, input_genes=None):
    # type: (Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig], Optional[List]) -> List[List[int, int, List[int]]]
    """Initialize gene list"""
    sce = SUScenario(cf)
    return sce.initialize_gene_list(input_gene_list=input_genes)


def scenario_effectiveness(cf, ind):
    # type: (Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig], array.array) -> array.array
    """Run SEIMS-based model and calculate economic and environmental effectiveness."""
    # 1. instantiate the inherited Scenario class.
    sce = SUScenario(cf)
    ind.id = sce.set_unique_id()
    setattr(sce, 'gene_values', ind)
    # 2. update BMP configuration units and related data according to gene_values,
    #      i.e., bmps_info and units_infos
    sce.boundary_adjustment()
    # 3. decode gene values to BMP items and exporting to MongoDB.
    sce.decoding()
    sce.export_to_mongodb()
    # 4. execute the SEIMS-based watershed model and get the timespan
    sce.execute_seims_model()
    ind.io_time, ind.comp_time, ind.simu_time, ind.runtime = sce.model.GetTimespan()
    # 5. calculate scenario effectiveness and delete intermediate data
    sce.calculate_economy()
    sce.calculate_environment()
    # 6. Export scenarios information
    sce.export_scenario_to_txt()
    sce.export_scenario_to_gtiff()
    # 7. Clean the intermediate data of current scenario
    # sce.clean(scenario_id=sce.ID, delete_scenario=True, delete_spatial_gfs=True)
    # 8. Assign fitness values
    ind.fitness.values = [sce.economy, sce.environment]

    return ind


def scenario_effectiveness_gene_list(cf, ind):
    # type: (Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig], array.array) -> array.array
    """Run SEIMS-based model and calculate economic and environmental effectiveness by period."""
    # 1. instantiate the inherited Scenario class.
    sce = SUScenario(cf)
    ind.id = sce.set_unique_id()
    if sce.check_valid_gene_list(ind):
        sce.initialize_gene_list(input_gene_list=ind)

    # 2. decode gene values to BMP items and exporting to MongoDB.
    sce.decoding_from_gene_list()
    sce.export_to_mongodb()

    # 3. first evaluate economic investment to exclude scenarios that don't satisfy the constraints
    # if that don't satisfy the constraints, don't execute the simulation process
    satisfied, [costs, maintains, incomes] = sce.satisfy_investment_plan()  # sce.check_custom_constraints():
    if not satisfied:
        sce.adjust_year_for_invest() # only when not satisfied
    else:
        # 4. execute the SEIMS-based watershed model and get the timespan
        sce.execute_seims_model()
        ind.io_time, ind.comp_time, ind.simu_time, ind.runtime = sce.model.GetTimespan()
        # 5. calculate scenario effectiveness and delete intermediate data
        sce.calculate_economy_by_period(costs, maintains, incomes)
        sce.calculate_environment_by_period()
    # 6. Export scenarios information
    sce.export_scenario_to_txt()
    sce.export_scenario_to_gtiff()
    # 7. Clean the intermediate data of current scenario
    # sce.clean(delete_scenario=True, delete_spatial_gfs=True)
    # 8. Assign fitness values
    ind.fitness.values = [sce.economy, sce.environment]
    ind.sed_sum = sce.sed_sum
    ind.sed_per_period = sce.sed_per_period
    ind.net_costs_per_period = sce.net_costs_per_period
    ind.costs_per_period = sce.costs_per_period
    ind.incomes_per_period = sce.incomes_per_period

    return ind

def scenario_objectives_gene_list(cf, ind):
    # type: (Union[SASlpPosConfig, SAConnFieldConfig, SACommUnitConfig], array.array) -> array.array
    """Run SEIMS-based model and calculate optimization objectives: economy, environment, maintenance."""
    # 1. instantiate the inherited Scenario class.
    sce = SUScenario(cf)
    ind.id = sce.set_unique_id()
    if sce.check_valid_gene_list(ind):
        sce.initialize_gene_list(input_gene_list=ind)

    # 2. decode gene values to BMP items and exporting to MongoDB.
    sce.decoding_from_gene_list()
    sce.export_to_mongodb()

    # 3. first evaluate economic investment to exclude scenarios that don't satisfy the constraints
    # if that don't satisfy the constraints, don't execute the simulation process
    satisfied, [costs, maintains, incomes] = sce.satisfy_investment_plan()  # sce.check_custom_constraints():
    # print(satisfied)
    if not satisfied:
        sce.adjust_year_for_invest() # only when not satisfied
    else:
        # 4. execute the SEIMS-based watershed model and get the timespan
        sce.execute_seims_model()
        ind.io_time, ind.comp_time, ind.simu_time, ind.runtime = sce.model.GetTimespan()
        # 5. calculate scenario effectiveness and delete intermediate data
        sce.calculate_economy_by_period(costs, maintains, incomes)
        sce.calculate_environment_by_period()
        sce.count_maintain_by_period()
    # 6. Export scenarios information
    sce.export_scenario_to_txt()
    sce.export_scenario_to_gtiff()
    # 7. Clean the intermediate data of current scenario
    # sce.clean(delete_scenario=True, delete_spatial_gfs=True)
    # 8. Assign fitness values
    ind.fitness.values = [sce.economy, sce.environment, sce.maintain_times]
    ind.sed_sum = sce.sed_sum
    ind.sed_per_period = sce.sed_per_period
    ind.net_costs_per_period = sce.net_costs_per_period
    ind.costs_per_period = sce.costs_per_period
    ind.incomes_per_period = sce.incomes_per_period
    ind.maintain_per_period = sce.maintain_per_period

    return ind

def main_single(sceid, gene_values):
    """Test of single evaluation of scenario."""
    cf = get_config_parser()
    base_cfg = SAConfig(cf)  # type: SAConfig
    if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        cfg = SASlpPosConfig(cf)
    elif base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
        cfg = SAConnFieldConfig(cf)
    else:  # Common spatial units, e.g., HRU and EXPLICITHRU
        cfg = SACommUnitConfig(cf)
    cfg.construct_indexes_units_gene()

    sce = SUScenario(cfg)
    sce.initialize(input_genes=gene_values)
    sce.boundary_adjustment()
    sce.set_unique_id(sceid)
    # print(sceid, sce.gene_values.__str__())
    sce.decoding()
    sce.export_to_mongodb()
    sce.execute_seims_model()
    sce.export_scenario_to_gtiff(sce.model.output_dir + os.sep + 'scenario_%d.tif' % sceid)
    sce.calculate_economy()
    sce.calculate_environment()

    print('Scenario %d: %s\n' % (sceid, ', '.join(repr(v) for v in sce.gene_values)))
    print('Effectiveness:\n\teconomy: %f\n\tenvironment: %f\n' % (sce.economy, sce.environment))

def main_multiple(eval_num):
    # type: (int) -> None
    """Test of multiple evaluations of scenarios."""
    cf = get_config_parser()
    base_cfg = SAConfig(cf)  # type: SAConfig
    if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        cfg = SASlpPosConfig(cf)
    elif base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
        cfg = SAConnFieldConfig(cf)
    else:  # Common spatial units, e.g., HRU and EXPLICITHRU
        cfg = SACommUnitConfig(cf)
    cfg.construct_indexes_units_gene()

    cost = list()
    for _ in range(eval_num):
        sce = SUScenario(cfg)
        sce.initialize()
        sceid = sce.set_unique_id()
        print(sceid, sce.gene_values.__str__())
        sce.calculate_economy()
        cost.append(sce.economy)
    print(max(cost), min(cost), sum(cost) / len(cost))


def main_manual(sceid, gene_values):
    """Test of set scenario manually (from input genes)."""
    cf = get_config_parser()
    base_cfg = SAConfig(cf)  # type: SAConfig
    if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        cfg = SASlpPosConfig(cf)
    elif base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
        cfg = SAConnFieldConfig(cf)
    else:  # Common spatial units, e.g., HRU and EXPLICITHRU
        cfg = SACommUnitConfig(cf)
    cfg.construct_indexes_units_gene()
    sce = SUScenario(cfg)

    sce.set_unique_id(sceid)
    sce.initialize(input_genes=gene_values)
    sce.boundary_adjustment()

    sce.decoding()
    sce.export_to_mongodb()
    sce.execute_seims_model()
    sce.export_sce_tif = True
    sce.export_scenario_to_gtiff(sce.model.output_dir + os.sep + 'scenario_%d.tif' % sceid)
    sce.calculate_economy()
    sce.calculate_environment()
    sce.export_sce_txt = True
    sce.export_scenario_to_txt()

    print('Scenario %d: %s\n' % (sceid, ', '.join(repr(v) for v in sce.gene_values)))
    print('Effectiveness:\n\teconomy: %f\n\tenvironment: %f\n\tsed_sum: %f\n' % (
        sce.economy, sce.environment, sce.sed_sum))
    return sceid, sce.economy, sce.environment, sce.sed_sum

    # sce.clean(delete_scenario=True, delete_spatial_gfs=True)


def main_manual_gene_list(sceid, input_gene_list = None):
    """Test of set scenario manually, with given scenario id and input gene list."""
    cf = get_config_parser()
    base_cfg = SAConfig(cf)  # type: SAConfig
    if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        cfg = SASlpPosConfig(cf)
    elif base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
        cfg = SAConnFieldConfig(cf)
    else:  # Common spatial units, e.g., HRU and EXPLICITHRU
        cfg = SACommUnitConfig(cf)
    cfg.construct_indexes_units_gene()
    sce = SUScenario(cfg)

    sce.set_unique_id(sceid)
    sce.initialize_gene_list(input_gene_list=input_gene_list)
    sce.decoding_from_gene_list()
    sce.export_to_mongodb()
    satisfied, [costs, maintains, incomes] = sce.satisfy_investment_plan()
    # print('investments: ', costs + maintains)
    if satisfied:
        sce.execute_seims_model()
        sce.calculate_economy_by_period(costs, maintains, incomes)
        
        sce.calculate_environment_by_period()
        print(f"Environment:{sce.environment}")
        sce.export_sce_tif = True
        sce.export_scenario_to_gtiff(sce.model.output_dir + os.sep + 'scenario_%d.tif' % sceid)
        sce.export_sce_txt = True
        sce.export_scenario_to_txt()

        print('Scenario %d: %s\n' % (sceid, ', '.join(repr(v) for v in sce.gene_values)))
        print('Effectiveness:\n\teconomy: %f\n\tenvironment: %f\n\tsed_sum: %f\n\t'
              'sed_per_period: %s\n\tnet_costs_per_period: %s\n\tcosts_per_period: %s\n\t'
              'incomes_per_period: %s'
              % (sce.economy, sce.environment, sce.sed_sum, str(sce.sed_per_period),
                 str(sce.net_costs_per_period), str(sce.costs_per_period),
                 str(sce.incomes_per_period)))

    # sce.clean(delete_scenario=True, delete_spatial_gfs=True)


def generate_tiff_txt(sceid, gene_values):
    cf = get_config_parser()
    base_cfg = SAConfig(cf)  # type: SAConfig
    if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        cfg = SASlpPosConfig(cf)
    elif base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
        cfg = SAConnFieldConfig(cf)
    else:  # Common spatial units, e.g., HRU and EXPLICITHRU
        cfg = SACommUnitConfig(cf)
    cfg.construct_indexes_units_gene()
    sce = SUScenario(cfg)

    sce.set_unique_id(sceid)
    sce.initialize(input_genes=gene_values)
    sce.decoding()
    sce.export_to_mongodb()
    # indicate the model has run
    sce.modelrun = True
    sce.modelout_dir = sce.model.output_dir
    # sce.calculate_economy()
    # sce.calculate_environment()
    sce.export_sce_tif = True
    sce.export_scenario_to_gtiff(sce.model.output_dir + os.sep + 'scenario_%d.tif' % sceid)
    sce.export_sce_txt = True
    sce.export_scenario_to_txt()

    print('Scenario %d: %s\n' % (sceid, ', '.join(repr(v) for v in sce.gene_values)))
    print('Effectiveness:\n\teconomy: %f\n\tenvironment: %f\n' % (sce.economy, sce.environment))

    # Not responsible for deleting
    # sce.clean(delete_scenario=True, delete_spatial_gfs=True)



if __name__ == '__main__':
    
    # main_multiple(3)
    # test with given gene list
    #input_gene_list =  [[0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 1, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 3, [0, 0, 0, 1, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 1, [0, 0, 1, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 1, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 0, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 2, [0, 0, 0, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 4, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 1, [0, 1, 1, 1, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 3, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 1, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 4, [0, 0, 0, 0, 1]], [2, 3, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 1, [0, 1, 1, 1, 1]], [1, 3, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [2, 5, [0, 0, 0, 0, 0]], [2, 5, [0, 0, 0, 0, 0]], [1, 5, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 2, [0, 0, 1, 1, 1]], [1, 4, [0, 0, 0, 0, 0]]]
    #sceid = 661042474
    #main_manual_gene_list(sceid, input_gene_list)

    #input_gene_list =  [[0, 0, [0, 0, 0, 0, 0]], [1, 3, [0, 0, 0, 1, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 1, 0]], [1, 5, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 5, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 5, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 0, 1]], [2, 5, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [1, 3, [0, 0, 0, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [1, 3, [0, 0, 0, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [1, 1, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 1, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 5, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 1, 1, 0]], [2, 4, [0, 0, 0, 0, 1]], [2, 1, [0, 1, 1, 1, 1]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 1, 1]], [2, 4, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [2, 3, [0, 0, 0, 0, 1]], [0, 0, [0, 0, 0, 0, 0]], [2, 2, [0, 0, 0, 0, 1]], [2, 1, [0, 0, 1, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [1, 1, [0, 0, 1, 0, 0]], [0, 0, [0, 0, 0, 0, 0]], [0, 0, [0, 0, 0, 0, 0]]]
    #sceid = 514658637
    #main_manual_gene_list(sceid, input_gene_list)

    main_manual_gene_list(1112)
   

# cf = get_config_parser()
# # cfg = SAConfig(cf)  # type: SAConfig
# cfg = SAConnFieldConfig(cf)
# sceobj = SUScenario(cfg)  # type: Scenario
#
# # test the picklable of Scenario class.
# import pickle
#
# s = pickle.dumps(sceobj)
# # print(s)
# new_cfg = pickle.loads(s)  # type: Scenario
# print(new_cfg.modelcfg.ConfigDict)
# print('Model time range: %s - %s' % (new_cfg.model.start_time.strftime('%Y-%m-%d %H:%M:%S'),
#                                      new_cfg.model.end_time.strftime('%Y-%m-%d %H:%M:%S')))
# print('model scenario ID: %d, configured scenario ID: %d' % (new_cfg.model.scenario_id,
#                                                              new_cfg.ID))
# new_cfg.set_unique_id()
# print('model scenario ID: %d, configured scenario ID: %d' % (new_cfg.model.scenario_id,
#                                                              new_cfg.ID))
