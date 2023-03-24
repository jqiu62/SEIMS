/*!
 * \file template.h
 * \brief Brief description of this module
 *        Detail description about the implementation.
 * \author Dawei Xiao
 * \date 2022-12-30
 *
 */
#ifndef SEIMS_MODULE_TEMPLATE_H
#define SEIMS_MODULE_TEMPLATE_H
//#define IS_DEBUG 0
#define IS_DEBUG 1
#include "SimulationModule.h"

using namespace std;

class CASC2D_OF : public SimulationModule {
public:
	CASC2D_OF(); //! Constructor

    ~CASC2D_OF(); //! Destructor

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

	void SetRasterPositionDataPointer(const char* key, int** positions) OVERRIDE;

	void OvrlDepth();

	void ChannDepth();

	void OvrlRout();

	float ovrl(int icell, int rbCell);

	void ChannRout();

	void chnchn(int reachIndex, int curReachId, int nReaches, int iCell, vector<int> vecCells);

	void RoutOutlet();

	float newChnDepth(float wch, float dch, float sfactor,
		int idCell, float addedVolume);

	float chnDischarge(float hchan, float hh, float wch, float dch,
		float stordep, float rmanch, float a, float sf, float sfactor);

	void printPosition();

	void CASC2D_OF::buildPositionIndex();

	void CASC2D_OF::printLineBreak(int lastrow_firstcol);
	void CASC2D_OF::printOvFlow();
	void CASC2D_OF::printChFlow();
	void CASC2D_OF::printCellFlow(int iCell);
	void CASC2D_OF::printCellArrow(int iCell, int lastCol, int curCol);
	void CASC2D_OF::printArrow(int iCell);
	void CASC2D_OF::deleteExistFile(string file);


private:
	int counter;
	std::ofstream  ovFlowFptr;
	std::ofstream  chFlowFptr;
	std::ofstream  position_Fptr;
	std::ofstream  wtrDepFptr;
	
    int m_nCells;
	int m_maxSoilLyrs;
	float* m_nSoilLyrs;
	float** m_ks;
	float* m_soilWtrStoPrfl;
	
	int m_nSubbsns;                             // subbasin number
	int m_inputSubbsnID;                    // current subbasin ID, 0 for the entire watershed
	vector<int> m_subbasinIDs;         // subbasin IDs
	clsSubbasins* m_subbasinsInfo;   // All subbasins information
	/// id of source cells of reaches
	int *m_sourceCellIds;
	/**
	*	The first element in each sub-array is the number of flow in cells in this sub-array
	*   �洢ÿ��դ��Ԫ��������Ԫ
	*   ����:
	*          [[2, 100,200],[3,101,201,202]]
	*          ��ʾ��һ��դ��Ԫ��������Ԫ��2����id�ֱ�Ϊ100��200���ڶ���դ��Ԫ��������Ԫ��3����id�ֱ�Ϊ101��201��202
	*/
	float **m_flowInIndex;
	/// map from subbasin id to index of the array
	map<int, int> m_idToIndex;

	/**** ������� ***/
	int m_nrows;					/* ����*/
	int m_ncols;						/* ����*/
	float m_dt;						/* dtʱ�䲽�� */												/* DT_HS �� DT_CH from FILE_IN*/
	float m_cellWth;				/* wդ��Ԫ�Ŀ��*/										/* Tag_CellWidth	from */
	float* m_ManningN;		/* pman[iman[][]]դ��Ԫ�ϵ�����ϵ��*/		/* VAR_MANNING from ParameterDB*/
	float* m_streamLink;		/* linkդ��Ԫ�ϵĺӵ����*/							/* VAR_STREAM_LINK from ParameterDB*/
	int m_nreach;					/* maxlink�ӵ�����*/
	float* m_flowOutIndex;	/* D8�����㷨��flow out index*/
	int m_idOutlet;					/* jout kout�����ˮ�ڵ�id*/								/* Tag_FLOWOUT_INDEX_D8*/
	float* m_surSdep;			/* sdep[i][j] դ��Ԫ�ϵĳ�ʼ��ˮ���*/				/* VAR_SUR_SDEP from ParameterDB, need initialize in preprocess*/
	map<int, vector<int> > m_reachLayers;	/* �ӵ�ͼ��*/							/* from reach propertites*/
	map<int, vector<int> > m_reachs;				/* */
	int** m_RasterPostion;		/* ����դ��λ�����ݵ�����*/
	map<int, int> m_downStreamReachId;	/* �����ϡ����κӵ�id��ӳ���ϵ*/
	float* m_chDownStream;	/* downstream id (The value is 0 if there if no downstream reach)*/
	/**** ����һά���� ***/
	float* m_surWtrDepth;		/* hդ��Ԫ�ϵĵر�ˮ��m*/								/* VAR_SURU from DepressionFS module */
	float* m_chDepth;				/* chp[i][j][3] դ��Ԫ�ϵĺӵ���� ȫ�ӵ�һ��ֵ*/
	float* m_chWidth;				/* chp[i][j][2] դ��Ԫ�ϵĺӵ����*/
	float* m_chSinuosity;			/* chp[i][j][6] դ��Ԫ�ϵĺӵ��������� ȫ�ӵ�һ��ֵ*/
	float* m_dem;						/* e դ��Ԫ�ϵĸ߳�*/
	float* m_chWtrDepth;			/* hch դ��Ԫ�ϵĺӵ�ˮ���ʼ��ʱ��ͬ��m_surWtrDepth����������ģ������*/
	float* m_Slope;					/* �¶�*/
	/**** ���һά���� ***/
	float* m_chQ;						/* dqov դ��Ԫ�ĵر������� ������/s*/
	float* m_ovQ;						/* dqch դ��Ԫ�ĺӵ��������� ������/s*/
	float m_outQ;						/* qout ��ˮ��դ��Ԫ�ĵر������� ������/s*/
	float m_outV;						/* vout ��ˮ��դ��Ԫ�ĵر������ ������*/

	float qoutov;						/* ��ˮ��դ��Ԫÿ��ʱ�䲽���ĳ����������Եر����� ������/s */	
	float sovout;						/* ��ˮ��դ��Ԫ���¶� */										
	/**** �������� ***/
	bool m_InitialInputs;						/* �Ƿ�����������*/
	map<int, vector<int>> m_rbcellsMap;  /* ���դ��Ԫ�ҡ��·��ĵ�Ԫ�±꣬keyդ��Ԫ��һά�����е��±꣬value[0]�ҷ�դ����±�(�����ҷ�դ������-1)��value[1]�·�դ����±꣨�����·�դ������-1��*/
	int output_icell;
	int printOvFlowMaxT;
	int printIOvFlowMinT;
	int printChFlowMaxT;
	int printChFlowMinT;
	int** m_RasterNeighbor;		/* ����դ��λ�����ݵ�����*/
	float ** m_Dqq;
	float * cellWtrDep;
};

#endif /* SEIMS_MODULE_TEMPLATE_H */
