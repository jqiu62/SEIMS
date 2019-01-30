/*!
 * \file Interpolate.h
 * \brief Interpolate function for site based data, e.g. precipitation
 *
 * Changelog:
 *   - 1. 2018-05-07 - lj - Code reformat.
 *
 * \author Junzhi Liu, Liangjun Zhu
 * \date Jan. 2010
 */
#ifndef SEIMS_MODULE_ITP_H
#define SEIMS_MODULE_ITP_H

#include "SimulationModule.h"

/** \defgroup ITP
 * \ingroup Climate
 * \brief Interpolation Module
 */
/*!
 * \class Interpolate
 * \ingroup ITP
 * \brief Interpolation
 */
class Interpolate: public SimulationModule {
public:
    Interpolate();

    ~Interpolate();

    void SetClimateDataType(float value) OVERRIDE;

    int Execute() OVERRIDE;

    void SetValue(const char* key, float value) OVERRIDE;

    void Set1DData(const char* key, int n, float* data) OVERRIDE;

    void Set2DData(const char* key, int n_rows, int n_cols, float** data) OVERRIDE;

    bool CheckInputData() OVERRIDE;

    void Get1DData(const char* key, int* n, float** data) OVERRIDE;

    /*!
     * \brief Check length of the input variable
     * \param[in] key the key to identify the requested data
     * \param[in] n size of the input 1D data
     * \param[out] m_n the corresponding member variable of length
     */
    static bool CheckInputSize(string& key, int n, int& m_n);


private:
    // This is the climate data type. It is used to get the specific lapse rate from lapse_rate table.
    // It is also used to create a string which can match the output id.
    // For example, if data_type = 1, i.e. the data type is P, main program will connect the output variable name "D"
    // and the data type to create a string like D_P,
    // this string is the same with the output id in the output lookup table and file.out.
    int m_dataType;
    /// count of stations
    int m_nStatioins;
    /// data of stations
    float* m_stationData;
    /// count of valid cells
    int m_nCells;
    /// weights of each sites of all valid cells
    float* m_itpWeights;

    /// whether using vertical interpolation
    bool m_itpVertical;
    /// elevation of stations
    float* m_hStations;
    /// elevation of cells
    float* m_dem;
    /// Lapse Rate, a 2D array. The first level is by month, and the second level is by data type in order of (P,T,PET).
    float** m_lapseRate;
    /// interpolation result
    float* m_itpOutput;
};
#endif /* SEIMS_MODULE_ITP_H */
