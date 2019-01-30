/*!
 * \file template.h
 * \brief Brief description of this module
 *        Detail description about the implementation.
 * \author Liangjun Zhu
 * \date 2018-02-07
 *
 * Changelog:
 *   - 1. 2018-02-07 - lj - Initial implementaition
 *   - 2. 2019-01-30 - lj - Add (or update) all available APIs
 */
#ifndef SEIMS_MODULE_TEMPLATE_H
#define SEIMS_MODULE_TEMPLATE_H

#include "SimulationModule.h"

using namespace std;

class ModuleTemplate: public SimulationModule {
public:
    ModuleTemplate(); //! Constructor

    ~ModuleTemplate(); //! Destructor

    ///////////// SetData series functions /////////////

    void SetValue(const char* key, float value) OVERRIDE;

    void SetValueByIndex(const char* key, int index, float value) OVERRIDE;

    void Set1DData(const char* key, int n, float* data) OVERRIDE;

    void Set2DData(const char* key, int n, int col, float** data) OVERRIDE;

    void SetReaches(clsReaches* rches) OVERRIDE;

    void SetSubbasins(clsSubbasins* subbsns) OVERRIDE;

    void SetScenario(Scenario* sce) OVERRIDE;

    ///////////// CheckInputData and InitialOutputs /////////////

    bool CheckInputData() OVERRIDE;

    void InitialOutputs() OVERRIDE;

    ///////////// Main control structure of execution code /////////////

    int Execute() OVERRIDE;

    ///////////// GetData series functions /////////////

    TimeStepType GetTimeStepType() OVERRIDE;

    void GetValue(const char* key, float* value) OVERRIDE;

    void Get1DData(const char* key, int* n, float** data) OVERRIDE;

    void Get2DData(const char* key, int* n, int* col, float*** data) OVERRIDE;

private:
    ///////////// Module specific functions /////////////

    /*!
    * \brief Check the input size of the first dimension of array-based data.
    *        Make sure all the input data have same first dimension.
    *
    * \param[in] key The key of the input data
    * \param[in] n The input data dimension
    * \return bool The validity of the dimension
    */
    bool CheckInputSize(const char *key, int n);

private:
    int m_nCells; ///< valid cells number
};

#endif /* SEIMS_MODULE_TEMPLATE_H */
