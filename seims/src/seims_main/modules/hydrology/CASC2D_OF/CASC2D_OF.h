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

	//void SetRasterRows(int rows) OVERRIDE;

	//void SetRasterCols(int cols) OVERRIDE;

	//void SetReachDepthData( FloatRaster* positions) OVERRIDE;

	void OvrlDepth();

	void ChannDepth();

	void OvrlRout();

	void ovrl(int icell, int rbCell);

	void ChannRout();

	void chnchn(int reachIndex, int curReachId, int nReaches, int iCell, vector<int> vecCells);

	void RoutOutlet();

	float newChnDepth(float wch, float dch, float sfactor,
		int idCell, float addedVolume);

	float chnDischarge(float hchan, float hh, float wch, float dch,
		float stordep, float rmanch, float a, float sf, float sfactor);


private:
    int m_nCells; ///< valid cells number
	int m_maxSoilLyrs;
	float* m_nSoilLyrs;
	float** m_ks;
	float* m_soilWtrStoPrfl;

	//! subbasin number
	int m_nSubbsns;
	//! current subbasin ID, 0 for the entire watershed
	int m_inputSubbsnID;
	//! subbasin IDs
	vector<int> m_subbasinIDs;
	//! All subbasins information
	clsSubbasins* m_subbasinsInfo;

	/// id of source cells of reaches
	/// �ӵ���
	int *m_sourceCellIds;
	/**
	*	@brief 2d array of flow in cells
	*
	*	The first element in each sub-array is the number of flow in cells in this sub-array
	*   �洢ÿ��դ��Ԫ��������Ԫ
	*   ����:
	*          [[2, 100,200],[3,101,201,202]]
	*          ��ʾ��һ��դ��Ԫ��������Ԫ��2����id�ֱ�Ϊ100��200���ڶ���դ��Ԫ��������Ԫ��3����id�ֱ�Ϊ101��201��202
	*/
	float **m_flowInIndex;

	/// map from subbasin id to index of the array
	map<int, int> m_idToIndex;
	//! reach depth data from SPATIAL collection
	//FloatRaster* m_reachDepth;
	//float* m_chDepth;

	/**** problems ***/
	/*
	1. �����ｵ��ʹ��������ģ�������VAR_EXCP����casc_2d��m_rintȴ��ֻ�۳���ֲ�������Ľ��꣬
		֮�����infil()�����Ӿ�������п۳���������ȣ���������跨����һ�£���Ҫʹ��VAR_EXCP��
		�ͱ���ɾ��infil()���������޸�h�ĸ�ֵ�߼���
	2. ��casc2d�У�h��ʼ��ʱ����sdep��֮��ÿ��ʱ�䲽�����ݽ���-����-�������и��£���seims�У�
		����ģ��ÿ��ʱ�䲽����������ر�����ȣ��ⰴ��˵Ӧ����h����ʱ���Բ�����Ҫsdep���������
		��ֱ��ʹ��ÿ��ʱ�䲽��������ģ������ĵر���
	*/
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
	float* m_chDownStream;/// downstream id (The value is 0 if there if no downstream reach)
	/**** ����һά���� ***/
	//float* m_exsPcp;				/* m_rintդ��Ԫ�ϵĹ��ؽ���ǿ�� mm/s */		/* VAR_EXCP from MUR_MR module*/
	float* m_surWtrDepth;		/* hդ��Ԫ�ϵĵر�ˮ��m*/								/* VAR_SURU from DepressionFS module */
	float* m_chDepth;				/* chp[i][j][3] դ��Ԫ�ϵĺӵ���� ȫ�ӵ�һ��ֵ*/
	float* m_chWidth;				/* chp[i][j][2] դ��Ԫ�ϵĺӵ����*/
	float* m_chSinuosity;			/* chp[i][j][6] դ��Ԫ�ϵĺӵ��������� ȫ�ӵ�һ��ֵ*/
	float* m_dem;						/* e դ��Ԫ�ϵĸ߳�*/
	float* m_chWtrDepth;			/* hch դ��Ԫ�ϵĺӵ�ˮ���ʼ��ʱ��ͬ��m_surWtrDepth����������ģ������*/
	//float *m_chManning;		/* chp[i][j][3] դ��Ԫ�ϵĺӵ����*/
	//float *m_chSSlope;			/* chp[i][j][3] դ��Ԫ�ϵĺӵ����*/
	float* m_Slope;					/* �¶�*/
	/**** ���һά���� ***/
	float* m_chQ;						/* dqov դ��Ԫ�ĵر������� ������/s*/				
	float m_outQ;						/* qout ��ˮ��դ��Ԫ�ĵر������� ������/s*/
	float m_outV;						/* vout ��ˮ��դ��Ԫ�ĵر������ ������*/
	/**** �������� ***/
	bool m_InitialInputs = true;				/* �Ƿ�����������*/

	/**** OverDepth.cʹ�� ***/
	int m, n;						/* ��ˮ���������� */
	float** m_rint;				/* m_rint[j][k],դ��Ԫ�ϵĹ��ؽ���ǿ�ȣ��۳������������� mm/s */		/* VAR_EXCP*/
	float dt;						/* ʱ�䲽�� */																		/* DT_HS �� DT_CH*/
	float** dqov;				/* dqov[j][k] դ��Ԫ�ĵر������� ������/s*/						/* VAR_CH_V*/
	float w;							/* դ��Ԫ�Ŀ��*/																/* Tag_CellWidth*/


	/**** OverRout.cʹ�� ***/
	float** h;								/* overland depth, ���ر�ˮ��*/									/* ������ģ�����*/
	int** iman;							/* iman[i][j] դ��Ԫ�϶�Ӧ��������������*/				/*VAR_MANNING*/
	int** link;								/* link[i][j] դ��Ԫ�ϵĺӵ����*/								/* VAR_STREAM_LINK*/
	int** node;							/* link[i][j] դ��Ԫ�ϵĽڵ���*/								/* ����Ҫ*/
	float* pman;							/* pman[l] ������������l��Ӧ������ϵ��*/					/*VAR_MANNING*/
	float** sdep;						/* sdep[i][j] դ��Ԫ�ϵĳ�ʼ��ˮ���*/						/* ����Ҫ*/
	float** e;								/* e[i][j] դ��Ԫ�϶�Ӧ�ĸ߳�ֵ*/								/* VAR_DEM*/
	float chp[50][100][7];			/* chp[i][j][k] ��link i,node j�ϵ�k��������ֵ��K= 1: channel type, 2: width, 3: depth, 4: side slope, 5: Manning's 'n', 6: sinuosity*/    /*REACH_SLOPE  GetReachesSingleProperty(REACH_SLOPE, &m_chSlope);*/
	float** hch;							/* �ӵ�ˮ����� */														/* CH_DEPTH*/


	/*ChannDepthʹ��*/
	int maxlink;							/* �ӵ�����*/																		/*�ο� CH_DW ģ����DiffusiveWave::Execute()�жԺӵ��ı���nReaches*/
	int *nchan_node;					/* �ӵ��еĽڵ�����*/														/*�ο� CH_DW ģ����DiffusiveWave::Execute()�жԽڵ�ı���vecCells.size() */
	int ichn[50][100][3];				/* ichn[i][j][k]�ӵ�i���ڵ�j�����к�, k = 1�к�, k = 2�к�*/	/*�ο� CH_DW ģ����DiffusiveWave::Execute()�жԽڵ�id��ȡֵvecCells[iCell] */
	float **dqch;						/* ʱ�䲽����դ��Ԫ�ĺӵ��������� ������ / s*/				/* VAR_CH_V*/


	/*RoutOutletʹ��*/
	int jout;								/* ��ˮ�ڵ��к�/*														/*����VAR_ID_OUTLET����*/
	int kout;								/* ��ˮ�ڵ��к�/*														/*����VAR_ID_OUTLET����*/

	float qoutov;						/* ��ˮ��դ��Ԫÿ��ʱ�䲽���ĳ����������Եر����� ������/s */	/*	�ڲ�����*/
	float sovout;						/* ��ˮ��դ��Ԫ���¶� */										/*�Զ���*/					
	float wchout;						/* ��ˮ��դ��Ԫ�ĺӵ���� */								/*�Զ���*/		
	float dchout;						/* ��ˮ��դ��Ԫ�ĺӵ���� */								/*�Զ���*/		
	float rmanout;						/* ��ˮ��դ��Ԫ������ϵ�� */								/*�Զ���*/		
	float sout;							/* ��ˮ��դ��Ԫ�ĺӴ��¶� */								/*�Զ���*/		
	float sfactorout;				
	float qout;							/* ��ˮ��դ��Ԫÿ��ʱ�䲽���ĳ�����  ������/s */	/*VAR_QRECH*/
	float *q;								/* �û��Զ���վ��ĳ�������*/								/*�Զ���*/

};

#endif /* SEIMS_MODULE_TEMPLATE_H */
