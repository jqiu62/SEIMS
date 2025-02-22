/*!
 * \file BMPArealStructFactory.h
 * \brief Areal struct BMP factory
 *
 * Changelog:
 *   - 1. 2017-07-13 - lj - Partially rewrite this class, Scenario data only read from MongoDB.
 *                          DataCenter will perform the data updating.
 *   - 2. 2017-11-29 - lj - Code style review.
 *   - 3. 2018-04-12 - lj - Code reformat.
 *
 * \author Huiran Gao, Liangjun Zhu
 */
#ifndef SEIMS_BMP_AREALSTRUCT_H
#define SEIMS_BMP_AREALSTRUCT_H

#include "tinyxml.h"
#include "basic.h"
#include "data_raster.hpp"

#include "BMPFactory.h"
#include "ParamInfo.h"

using namespace ccgl;
using namespace bmps;

namespace bmps {
/*!
 * \class bmps::BMPArealStruct
 * \brief Manage areal Structural BMP data, inherited from ParamInfo
 */
class BMPArealStruct: Interface {
public:
    //! Constructor
    BMPArealStruct(const bson_t*& bsonTab, bson_iter_t& iter);
    //! Destructor
    ~BMPArealStruct();
    //! Get name
    string getBMPName() { return m_name; }
    //! Get suitable landuse
    vector<int>& getSuitableLanduse() { return m_landuse; }
    //! Get parameters
    map<string, ParamInfo<FLTPT>*>& getParameters() { return m_parameters; }
    ////! Is EffectivenessVariable
    //bool isEffectivenessVariable(){ return m_effectivenessVariable; }
    ////! get change frequency
    //int getChangeFrequency(){ return m_changeFrequency; }
    //! getter and setter for last update time
    time_t getLastUpdateTime() const { return m_lastUpdateTime; }
    void setLastUpdateTime(time_t val) { m_lastUpdateTime = val; }
private:
    int m_id;      ///< unique BMP ID
    string m_name; ///< name
    string m_desc; ///< description
    string m_refer; ///< references
    vector<int> m_landuse; ///< suitable placement landuse

    ////! Is BMP effectiveness variable or not
    //bool m_effectivenessVariable;
    ////! Set the change frequency in seconds, if the BMP effectiveness is variable
    //int m_changeFrequency;
    //! last update time of BMP effectiveness
    time_t m_lastUpdateTime;
    /*!
     * \key the parameter name, remember to add subbasin number as prefix when use GridFS file in MongoDB
     * \value the ParamInfo class
     */
    map<string, ParamInfo<FLTPT>*> m_parameters;
};

/*!
 * \class bmps::BMPArealStructFactory
 * \brief Initiate Areal Structural BMPs
 *
 */
class BMPArealStructFactory: public BMPFactory {
public:
    /// Constructor
    BMPArealStructFactory(int scenarioId, int bmpId, int subScenario,
                          int bmpType, int bmpPriority, vector<string>& distribution,
                          const string& collection, const string& location, bool effectivenessChangeable = false,
                          time_t changeFrequency = -1, int variableTimes = -1, float mtEffect = 0.f);

    /// Destructor
    ~BMPArealStructFactory();

    //! Load BMP parameters from MongoDB
    void loadBMP(MongoClient* conn, const string& bmpDBName) OVERRIDE;

    //! Set raster data if needed
    void setRasterData(map<string, IntRaster*>& sceneRsMap) OVERRIDE;

    //! Get management fields data
    int* GetRasterData() OVERRIDE { return m_mgtFieldsRs; }

    //! Get effect unit IDs
    const vector<int>& getUnitIDs() const { return m_unitIDs; }
    const vector<int>& getUnitIDsByIndex(){ return m_unitIDsSeries[m_seriesIndex]; }
    const map<int, int>& getUpdateTimesByIndex(){ return m_unitUpdateTimes[m_seriesIndex]; }

    //also maintain ids for each year
    const vector<int>& getUnitIDsforMTByIndex() { return m_unitMaintainRecords[m_seriesIndex]; }
    void increaseSeriesIndex(){ m_seriesIndex++; }
    int getSeriesIndex() { return m_seriesIndex; }
    float getMTEffect() { return m_mtEffect; }

    //! Get areal BMP parameters
    const map<int, BMPArealStruct*>& getBMPsSettings() const { return m_bmpStructMap; }

    //! Output
    void Dump(std::ostream* fs) OVERRIDE;

private:
    //! management units file name
    string m_mgtFieldsName;
    //! management units raster data
    int* m_mgtFieldsRs;
    //! locations
    vector<int> m_unitIDs;

    //! Store the spatial unit IDs that need to update for different years
    //! If the unit has been configured with bmp since some time, for years later on it will remain.
    vector<vector<int> > m_unitIDsSeries;
    //! How many times are the above spatial units updated respectively
    //! store together by year, store as {unit id: actual implemented years}
    vector<map<int,int> > m_unitUpdateTimes;
    //! Store unit ids for the year when maintenance is adpoted
    vector<vector<int>> m_unitMaintainRecords;
    float m_mtEffect;
    int m_seriesIndex;
    /*!
     *\key The unique areal BMP ID
     *\value Instance of BMPArealStruct
     */
    map<int, BMPArealStruct*> m_bmpStructMap;
};
}
#endif /* SEIMS_BMP_AREALSTRUCT_H */
