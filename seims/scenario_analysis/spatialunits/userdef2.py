"""User defined operation for optimizing BMPs based on slope position units.

    @author   : Liangjun Zhu, Huiran Gao

    @changelog:
    - 16-11-08  - hr - initial implementation.
    - 17-08-18  - lj - reorganization.
    - 18-02-09  - lj - compatible with Python3.
    - 18-11-07  - lj - support multiple BMPs configuration methods.
    - 18-12-04  - lj - add func:`crossover_updown` according to Wu et al. (2018).
    - 23-06-30  - adopted for gene list
"""
from __future__ import absolute_import, unicode_literals

from future.utils import viewitems
import array
from collections import OrderedDict
import os
import sys
import random
import math

from pygeoc.utils import get_config_parser, MathClass
from typing import List, Tuple, Dict, Union, Any, Optional, AnyStr

if os.path.abspath(os.path.join(sys.path[0], '../..')) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '../..')))

from scenario_analysis import _DEBUG
from scenario_analysis.spatialunits.scenario2 import select_potential_bmps


def check_individual_diff(old_ind,  # type: Union[array.array, List[int], Tuple[int]]
                          new_ind  # type: Union[array.array, List[int], Tuple[int]]
                          ):
    # type: (...) -> bool
    """Check the gene values of two individuals."""
    diff = False
    for i in range(len(old_ind)):
        if not MathClass.floatequal(old_ind[i], new_ind[i]):
            diff = True
            break
    return diff

def check_individual_diff_gene_list(old_ind,  # type: Union[array.array, List[int], Tuple[int]]
                          new_ind  # type: Union[array.array, List[int], Tuple[int]]
                          ):
    # type: (...) -> bool
    """Check whether the gene lists of two individuals are identical."""
    diff = False
    for i in range(len(old_ind)):
        if old_ind[i][0] != new_ind[i][0] or old_ind[i][1] != new_ind[i][1] or any(old_ind[i][2][ii] != new_ind[i][2][ii] for ii in range(0, len(old_ind[i][2]))):
            diff = True
            break
    return diff


#                                       #
#          Crossover (Mate)             #
#                                       #
# crossover_slppos_gene_list, crossover_updown_gene_list, crossover_rdm_gene_list

def crossover_slppos_gene_list(ind1,  # type: Union[array.array, List[int], Tuple[int]]
                     ind2,  # type: Union[array.array, List[int], Tuple[int]]
                     hillslp_values_num  # type: int
                     ):
    # type: (...) -> Tuple(Union[array.array, List[int], Tuple[int]], Union[array.array, List[int], Tuple[int]])
    """Crossover operator based on slope position units.
    Each individual can keep the domain knowledge based rules after crossover operation.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        hillslp_values_num: Gene values number of each hillslope.

    Returns:
        A tuple of two individuals.
    """
    size = min(len(ind1), len(ind2))
    assert (size > hillslp_values_num * 2)

    while True:
        cxpoint1 = random.randint(0, size - 1)
        cxpoint2 = random.randint(1, size)
        if cxpoint2 < cxpoint1:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        # print(cxpoint1, cxpoint2)
        cs1 = cxpoint1 // hillslp_values_num
        cs2 = cxpoint2 // hillslp_values_num
        # print(cs1, cs2)
        cxpoint1 = cs1 * hillslp_values_num
        if cxpoint2 % hillslp_values_num != 0:
            cxpoint2 = hillslp_values_num * (cs2 + 1)
        if cxpoint1 == cxpoint2:
            cxpoint2 += hillslp_values_num
        # print(cxpoint1, cxpoint2)
        if not (cxpoint1 == 0 and cxpoint2 == size):
            break  # avoid change the entire genes
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2


def crossover_updown_gene_list(ind1,  # type: Union[array.array, List[int], Tuple[int]]
                     ind2,  # type: Union[array.array, List[int], Tuple[int]]
                     updownunits,  # type: Dict[int, Dict[AnyStr, List[int]]]
                     gene2unit,  # type: Dict[int, int]
                     unit2gene  # type: OrderedDict[int, int]
                     ):
    """Crossover operator based on hydrologically connected fields with
    upstream-downstream relationships.

    A subtree exchange method introduced in Wu et al. (2018) is adapted and extended.

    - 1. A node (gene) is chosen randomly.
    - 2. If the pattern of 'downslope-current' of each individual after
         crossover is accord with the UPDOWN rules, then exchange the subtrees with
         the selected gene as root node.
    - 3. If not, check the downslope gene one by one, until a eligible gene is reached.
    - 4. If no eligible gene is found until the last gene is reached:
      - 4.1. If the number of the subtree with the last gene as root equals to all genes, return;
      - 4.2. Else, exchange the subtree like step 2.
    """

    def check_validation(geneidx, genevalue, ind):
        """Check if the gene value is valid according to UPDOWN method."""
        if geneidx <= 0:
            return True
        unitid = gene2unit[geneidx]
        downids = updownunits[unitid]['downslope']
        valid = True
        if _DEBUG:
            print('---- Check validation of crossover gene %d (field ID %d)'
                  ' with value %s' % (geneidx, unitid, repr(genevalue)))
        for downid in downids:
            if downid <= 0:
                if _DEBUG:
                    print('----   The field unit is the most downstream unit.')
                continue
            downgeneidx = unit2gene[downid]
            downgenevalue = ind[downgeneidx][0] #using gene list
            if _DEBUG:
                print('----   Downstream field ID: %d, gene value: %s' % (downid,
                                                                          repr(downgenevalue)))
            if downgenevalue > 0 and genevalue > 0:
                # Only when the planned gene and its downslope gene are both configured BMP,
                #   is considered invalid.
                valid = False
                break
        if _DEBUG:
            print('---- Validation check %s!' % ('PASSED' if valid else 'FAILED'))
        return valid

    def check_validation_individual(ind):
        """Check the validation of the individual."""
        valid = True
        for k, v in viewitems(updownunits):
            if isinstance(v['downslope'], list) and len(v['downslope']) == 0:
                continue
            if isinstance(v['downslope'], list):
                downslope_id = v['downslope'][0]
            else:
                downslope_id = v['downslope']
            if downslope_id <= 0:
                continue
            downslope_value = ind[unit2gene[downslope_id]][0]
            if ind[unit2gene[k]][0] > 0 and downslope_value > 0:
                valid = False
                if _DEBUG:
                    print('---- Field ID %d with value %s is invalid!' % (k,
                                                                          repr(ind[unit2gene[k]][0])))
                    print('----   The downstream field ID %d is with %s' % (downslope_id,
                                                                            repr(downslope_value)))
                return valid
        return valid

    if _DEBUG:
        print('-- Before crossover operation:')
        print('-- Individual 1 validation check %s' % ('PASSED' if check_validation_individual(ind1)
                                                       else 'FAILED'))
        print('-- Individual 2 validation check %s' % ('PASSED' if check_validation_individual(ind2)
                                                       else 'FAILED'))
    size = min(len(ind1), len(ind2))
    # 1. Chose a node (gene) randomly.
    cxpoint = random.randint(0, size - 1)
    if _DEBUG:
        print('-- Randomly selected crossover gene index: %d, field ID %d' % (cxpoint,
                                                                              gene2unit[cxpoint]))
    # 2. Whether it is valid after crossover
    untested = [cxpoint]
    while len(untested) > 0:
        cxp = untested[0]
        if check_validation(cxp, ind2[cxp][0], ind1) and check_validation(cxp, ind1[cxp][0], ind2):
            cxpoint = cxp
            break
        else:
            untested.remove(cxp)
            for i in updownunits[gene2unit[cxp]]['downslope']:
                if i > 0:
                    untested.append(unit2gene[i])
    if cxpoint < 0:
        return ind1, ind2
    upids = updownunits[gene2unit[cxpoint]]['all_upslope']
    if gene2unit[cxpoint] not in upids:
        upids.insert(0, gene2unit[cxpoint])  # contain the crossover point itself

    if len(upids) >= len(gene2unit):  # avoid exchange the entire genes
        if _DEBUG:
            print('-- No need to crossover since the entire genes is selected!')
        return ind1, ind2
    if len(upids) >= 1:
        same_subtree = True
        for sub in upids:
            if ind1[unit2gene[sub]][0] != ind2[unit2gene[sub]][0]:
                same_subtree = False
                break
        if _DEBUG and same_subtree:
            print('-- No need to crossover since the gene values of the subtree'
                  ' with the same values is selected!')
        if same_subtree:
            return ind1, ind2
    if _DEBUG:
        print('-- Adjusted crossover field ID: %d,'
              ' exchanged length: %d,'
              ' exchange field IDs: %s' % (gene2unit[cxpoint], len(upids), upids.__str__()))
    for upunitid in upids:
        upgeneidx = unit2gene[upunitid]
        tmplist = ind1[upgeneidx]
        ind1[upgeneidx] = ind2[upgeneidx]
        ind2[upgeneidx] = tmplist
    if _DEBUG:
        print('-- After crossover operation:')
        print('-- Individual 1 validation check %s' % ('PASSED' if check_validation_individual(ind1)
                                                       else 'FAILED'))
        print('-- Individual 2 validation check %s' % ('PASSED' if check_validation_individual(ind2)
                                                       else 'FAILED'))
    return ind1, ind2


def crossover_rdm_gene_list(ind1,  # type: Union[array.array, List[int], Tuple[int]]
                  ind2  # type: Union[array.array, List[int], Tuple[int]]
                  ):
    # type: (...) -> tuple(Union[array.array, List[int], Tuple[int]], Union[array.array, List[int], Tuple[int]])
    """Crossover randomly.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.

    Returns:
        A tuple of two individuals.
    """
    size = min(len(ind1), len(ind2))

    while True:
        cxpoint1 = random.randint(0, size - 1)
        cxpoint2 = random.randint(1, size)
        if cxpoint2 < cxpoint1:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        else:
            cxpoint2 += 1
        if not (cxpoint1 == 0 and cxpoint2 == size):
            break  # avoid change the entire genes
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2


#                                       #
#               Mutate                  #
#                                       #
# mutate_rdm_subscenario, mutate_rule_subscenario, mutate_implement_year, mutate_maintain

def mutate_rdm_gene_list( ind,  # type: Union[array.array, List[int], Tuple[int]]
                percent,  # type: float
                prob,  # type: float
                possible_gene,  # type: Union[List[int], Tuple[int]]
                ):
    # type: (...) -> Tuple(Union[array.array, List[int], Tuple[int]], Union[array.array, List[int], Tuple[int]])
    """
    Mutation Gene List randomly, in which, subscenario,implementation year and maintain list can be mutatable.
    """
    if percent > 0.5:
        percent = 0.5
    elif percent < 0.01:
        percent = 0.01
    try:
        mut_num = random.randint(1, int(len(ind) * percent))
    except ValueError or Exception:
        return ind
    muted = dict()

    for m in range(mut_num):
        if random.random() > prob:
            continue
        mpoint = random.randint(0, len(ind) - 1)
        while mpoint in muted.keys():
            mpoint = random.randint(0, len(ind) - 1)
        mplace = random.randint(0,2) # assign mutation to subscenario, implement, or maintain
        muted[mpoint] = mplace
        if mplace == 0:
            mutate_rdm_subscenario(ind, mpoint, possible_gene)
        elif mplace == 1:
            mutate_implement_year(ind, mpoint,len(ind[mpoint][2]))
        else:
            mutate_maintain(ind, mpoint, len(ind[mpoint][2]))
    return ind

def mutate_rule_gene_list(ind,  # type: Union[array.array, List[int], Tuple[int]]
                percent,  # type: float
                prob,  # type: float
                unitsinfo,  # type: Dict[Union[str, int], Any]
                gene2unit,  # type: Dict[int, int]
                unit2gene,  # type: OrderedDict[int, int]
                suitbmps,  # type: Dict[int, List[int]]
                unit='SLPPOS',  # type: AnyStr
                method='SUIT',  # type: AnyStr
                bmpgrades=None,  # type: Optional[Dict[int, int]]
                tagnames=None,  # type: Optional[List[Tuple[int, AnyStr]]] # Slope position units
                thresholds=None  # type: Optional[List[float]] # Only for slope position
                ):
    """
    Mutation Gene List based on rules for subscenario, while implementation year and maintain list can mutate randomly.
    """
    if percent > 0.5:
        percent = 0.5
    elif percent < 0.01:
        percent = 0.01
    try:
        mut_num = random.randint(1, int(len(unit2gene) * percent))
    except ValueError or Exception:
        return ind
    muted = dict()
    unit2gene_list = list(viewitems(unit2gene))
    ind_as_gene_values = [ind[i][0] for i in range(0, len(ind))]

    for m in range(mut_num):
        if random.random() > prob:
            continue
        mpoint = random.randint(0, len(ind) - 1)
        while mpoint in muted.keys():
            mpoint = random.randint(0, len(ind) - 1)
        mplace = random.randint(0,2) # assign mutation to subscenario, implement, or maintain
        if mplace == 1:
            mutate_implement_year(ind, mpoint,len(ind[mpoint][2]))
            muted[mpoint] = mplace
        elif mplace == 2:
            mutate_maintain(ind, mpoint, len(ind[mpoint][2]))
            muted[mpoint] = mplace
        else:
            unitid = unit2gene_list[mpoint][0]
            geneidx = unit2gene_list[mpoint][1]
            oldgenev = ind[geneidx][0]
            # begin to mutate on unitid
            # get the potential BMP IDs
            bmps = select_potential_bmps(unitid, suitbmps, unitsinfo, unit2gene, ind_as_gene_values,
                                         unit=unit, method=method,
                                         bmpgrades=bmpgrades, tagnames=tagnames)
            if bmps is None or len(bmps) == 0:
                continue
            # Get new BMP ID for current unit.
            if 0 not in bmps:
                bmps.append(0)
            if oldgenev in bmps:
                bmps.remove(oldgenev)
            if len(bmps) > 0:
                ind[geneidx][0] = bmps[random.randint(0, len(bmps) - 1)]
                if ind[geneidx][0] == 0:
                    ind[geneidx][1] = 0
                    ind[geneidx][2] = [0] * len(ind[geneidx][2])
                muted[mpoint] = mplace
            else:  # No available BMP
                pass
    # Step 2: Mutate on thresholds of boundary adaptive
    #         Only available for Slope position units
    if unit == 'SLPPOS' and len(ind) > len(unit2gene):
        muted_thresh = list()
        potmute_threshs = list()  # potential gene index for threshold mutation
        for tagidx, (tag, tagname) in enumerate(tagnames):
            for spid in unitsinfo[tagname]:
                if ind[unit2gene[spid]][0] <= 0:
                    continue  # No BMP configured
                if tagidx == 0:
                    potmute_threshs.append(unit2gene[spid] + len(tagnames))
                elif 0 < tagidx < len(tagnames) - 1:
                    potmute_threshs.append(unit2gene[spid] + len(tagnames) - 1)
                    potmute_threshs.append(unit2gene[spid] + len(tagnames))
                else:
                    potmute_threshs.append(unit2gene[spid] + len(tagnames) - 1)
        potmute_threshs = list(set(potmute_threshs))
        boundarymut_num = random.randint(0, int(len(potmute_threshs) * percent))
        if _DEBUG and boundarymut_num > 0:
            print('  Max mutate num for boundary adaptive: %d' % boundarymut_num)
        for m in range(boundarymut_num):
            if random.random() > prob:
                continue
            # mutate will happen
            mpoint = random.randint(0, len(potmute_threshs) - 1)
            while potmute_threshs[mpoint] in muted_thresh:
                mpoint = random.randint(0, len(potmute_threshs) - 1)
            newgenevs = thresholds[:]
            oldgenev = ind[potmute_threshs[mpoint]][0]
            if oldgenev in newgenevs:
                newgenevs.remove(oldgenev)
            if len(newgenevs) == 0:
                continue
            ind[potmute_threshs[mpoint]][0] = newgenevs[random.randint(0, len(newgenevs) - 1)]
            muted_thresh.append(potmute_threshs[mpoint])
            if _DEBUG:
                print('  Mutate on thresh index: %d, oldgene: %s, potThresh: %s, new gene: %s' %
                      (potmute_threshs[mpoint], repr(oldgenev), newgenevs.__str__(),
                       repr(ind[potmute_threshs[mpoint]][0])))
    return ind



def mutate_rdm_subscenario(ind,  # type: Union[array.array, List[int], Tuple[int]]
               index, possible_gene  # type: Union[List[int], Tuple[int]]
               ):
    # type: (...) -> Tuple(Union[array.array, List[int], Tuple[int]], Union[array.array, List[int], Tuple[int]])
    """
    Mutation Gene values at given index randomly, old gene value is excluded from target values.
    """
    if 0 not in possible_gene:
        possible_gene.append(0)
    target = possible_gene[:]
    target = list(set(target))
    if ind[index][0] in target:
        target.remove(ind[index][0])
    ind = random.randint(0, len(target) - 1)
    ind[index][0] = target[ind]
    if ind[index][0] == 0: # empty implement, maintain if any
        ind[index][1] = 0
        ind[index][2] = [0] * len(ind[index][2])
    return ind


def mutate_implement_year(ind, index, up):
    # mutate the implementation year with given index then adjust maintenance
    subscenario, impl_year, mt = ind[index]
    if subscenario == 0:
        return
    new_impl_year = random.randint(1, up)
    while impl_year == new_impl_year:
        new_impl_year = random.randint(1, up)
    mt = [0]*len(mt) # set all maintain to 0
    for i in range(new_impl_year, up):
        mt[i] = random.randint(0, 1) 
    ind[index] = [subscenario, new_impl_year, mt]

    return ind

def mutate_maintain(ind, index, up):
    # mutate the maintenance activities with given index
    subscenario, impl_year, mt = ind[index]
    if subscenario == 0:
        return
    if impl_year == up:
        return
    new_mt = mt[:]
    isSame = new_mt == mt
    while isSame:
        for i in range(impl_year, up):
            new_mt[i] = random.randint(0, 1)
        isSame = new_mt == mt
    ind[index] = [subscenario, impl_year, new_mt]

    return ind

def main_test_crossover_mutate_gene_list(gen_num, cx_rate, mut_perc, mut_rate):
    # type: (int, float, float, float) -> None
    """Test mutate function."""
    from deap import base
    from scenario_analysis import BMPS_CFG_UNITS, BMPS_CFG_METHODS
    from scenario_analysis.config import SAConfig
    from scenario_analysis.spatialunits.config import SASlpPosConfig, SAConnFieldConfig, \
        SACommUnitConfig
    from scenario_analysis.spatialunits.scenario2 import SUScenario
    cf = get_config_parser()

    base_cfg = SAConfig(cf)  # type: SAConfig
    if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:  # SLPPOS
        cfg = SASlpPosConfig(cf)
    elif base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[2]:  # CONNFIELD
        cfg = SAConnFieldConfig(cf)
    else:  # Common spatial units, e.g., HRU and EXPLICITHRU
        cfg = SACommUnitConfig(cf)
    cfg.construct_indexes_units_gene()

    # Initialize gene values for individual 1 and 2
    sce1 = SUScenario(cfg)
    ind1 = sce1.initialize_gene_list()
    sceid1 = sce1.set_unique_id()
    print('Scenario1-ID: %d\n  Initial genes: %s' % (sceid1, ind1.__str__()))
    sce2 = SUScenario(cfg)
    ind2 = sce2.initialize_gene_list()
    sceid2 = sce2.set_unique_id()
    print('Scenario2-ID: %d\n  Initial genes: %s' % (sceid2, ind2.__str__()))

    # Clone old individuals
    toolbox = base.Toolbox()

    for gen_id in list(range(gen_num)):
        # Calculate initial economic benefit
        inicost1 = sce1.calculate_economy()
        inicost2 = sce2.calculate_economy()
        print('## Generation %d ##' % gen_id)
        # Crossover
        print('Crossover:')
        old_ind1 = toolbox.clone(ind1)
        old_ind2 = toolbox.clone(ind2)
        if random.random() <= cx_rate:
            if base_cfg.bmps_cfg_method == BMPS_CFG_METHODS[3]:  # SLPPOS method
                crossover_slppos_gene_list(ind1, ind2, sce1.cfg.hillslp_genes_num)
            elif base_cfg.bmps_cfg_method == BMPS_CFG_METHODS[2]:  # UPDOWN method
                crossover_updown_gene_list(ind1, ind2, sce1.cfg.updown_units, sce1.cfg.gene_to_unit,
                                 sce1.cfg.unit_to_gene)
            else:
                crossover_rdm_gene_list(ind1, ind2)
        if not check_individual_diff_gene_list(old_ind1, ind1) and not check_individual_diff_gene_list(old_ind2, ind2):
            print('  No crossover occurred on Scenario1 and Scenario2!')
        else:
            print('  Crossover genes:\n    Scenario1: %s\n'
                  '    Scenario2: %s' % (ind1.__str__(), ind2.__str__()))
        # Calculate economic benefit after crossover
        sce1.initialize_gene_list(ind1)
        cxcost1 = sce1.calculate_economy()
        sce2.initialize_gene_list(ind2)
        cxcost2 = sce2.calculate_economy()

        # Mutate
        print('Mutate:')
        old_ind1 = toolbox.clone(ind1)
        old_ind2 = toolbox.clone(ind2)
        if base_cfg.bmps_cfg_method == BMPS_CFG_METHODS[0]:
            possible_gene_values = list(sce1.bmps_params.keys())
            if 0 not in possible_gene_values:
                possible_gene_values.append(0)
            mutate_rdm_gene_list(ind1, mut_perc, mut_rate, possible_gene_values)
            mutate_rdm_gene_list(ind2, mut_perc, mut_rate, possible_gene_values)
        else:
            tagnames = None
            if base_cfg.bmps_cfg_unit == BMPS_CFG_UNITS[3]:
                tagnames = sce1.cfg.slppos_tagnames
            mutate_rule_gene_list(ind1, mut_perc, mut_rate, sce1.cfg.units_infos, sce1.cfg.gene_to_unit, sce1.cfg.unit_to_gene,
                        sce1.suit_bmps['LANDUSE'],
                        unit=base_cfg.bmps_cfg_unit, method=base_cfg.bmps_cfg_method,
                        bmpgrades=sce1.bmps_grade, tagnames=tagnames,
                        thresholds=sce1.cfg.boundary_adaptive_threshs)
            mutate_rule_gene_list(ind2, mut_perc, mut_rate, sce2.cfg.units_infos, sce2.cfg.gene_to_unit, sce2.cfg.unit_to_gene,
                        sce2.suit_bmps['LANDUSE'],
                        unit=base_cfg.bmps_cfg_unit, method=base_cfg.bmps_cfg_method,
                        bmpgrades=sce2.bmps_grade, tagnames=tagnames,
                        thresholds=sce2.cfg.boundary_adaptive_threshs)
        if not check_individual_diff_gene_list(old_ind1, ind1):
            print('  No mutation occurred on Scenario1!')
        else:
            print('    Mutated genes of Scenario1: %s' % ind1.__str__())
        if not check_individual_diff_gene_list(old_ind2, ind2):
            print('  No mutation occurred on Scenario2!')
        else:
            print('    Mutated genes of Scenario2: %s' % ind2.__str__())

        # Calculate economic benefit after mutate
        sce1.initialize_gene_list(ind1)
        mutcost1 = sce1.calculate_economy()
        sce2.initialize_gene_list(ind2)
        mutcost2 = sce2.calculate_economy()

        print('Initial cost: \n'
              '  Scenario1: %.3f, Scenario2: %.3f\n'
              'Crossover:\n'
              '  Scenario1: %.3f, Scenario2: %.3f\n'
              'Mutation:\n'
              '  Scenario1: %.3f, Scenario2: %.3f\n' % (inicost1, inicost2,
                                                        cxcost1, cxcost2,
                                                        mutcost1, mutcost2))



if __name__ == '__main__':
    main_test_crossover_mutate_gene_list(4, 0.8, 0.2, 0.1)
