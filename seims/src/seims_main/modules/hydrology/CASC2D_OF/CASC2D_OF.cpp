#include "CASC2D_OF.h"
#include "text.h"

CASC2D_OF::CASC2D_OF() :
    m_nCells(-1),m_nSoilLyrs(nullptr),m_ks(nullptr),m_soilWtrStoPrfl(nullptr),
	m_surWtrDepth(nullptr), m_chWtrDepth(nullptr),m_surSdep(nullptr), m_ManningN(nullptr), m_streamLink(nullptr), m_dem(nullptr) ,
	m_flowOutIndex(nullptr) , m_Slope(nullptr), m_chWidth(nullptr) , m_chSinuosity(nullptr), m_chQ(nullptr), m_outQ(0.0), m_outV(0.0){

}

CASC2D_OF::~CASC2D_OF() {
}

// set 
void CASC2D_OF::SetValue(const char* key, float value) {
	string sk(key);
	if (StringMatch(sk, Tag_HillSlopeTimeStep)) {
		m_dt = value;
	}else if (StringMatch(sk, Tag_CellWidth)) {
		m_cellWth = value;
	}else if (StringMatch(sk, HEADER_RS_NROWS)){
		m_nrows = value;
	}else if (StringMatch(sk, HEADER_RS_NCOLS))
	{
		m_ncols = value;
	}
}

void CASC2D_OF::SetValueByIndex(const char* key, int index, float value) {
}

void CASC2D_OF::Set1DData(const char* key, int n, float* data) {
	/* todo ���ⲿ���յĽ���ǿ��һάת��ά����ֵ��m_rint** */
	if (!CheckInputSize("CASC2D_OF", key, n, m_nCells)) return;
	string sk(key);
	 if (StringMatch(sk, VAR_SURU)) {
		m_surWtrDepth = data;  
	}
	// else if (StringMatch(sk, VAR_SUR_SDEP)) {
	//	m_surSdep = data;
	//}
	else if (StringMatch(sk, VAR_MANNING)){
		m_ManningN = data;
	}else if (StringMatch(sk, VAR_STREAM_LINK)) {
		m_streamLink = data;
		//for (size_t i = 0; i < 300000; i++)
		//{
		//	if (m_streamLink[i] > 0)
		//	{
		//		cout << i << ":" << m_streamLink[i] << endl;
		//	}
		//}
	}else if (StringMatch(sk, VAR_DEM)) {
		m_dem = data;
	}else if (StringMatch(sk, Tag_FLOWOUT_INDEX_D8)) {
		m_flowOutIndex = data;
		for (int i = 0; i < m_nCells; i++) {
			if (m_flowOutIndex[i] < 0) {
				m_idOutlet = i;
				break;
			}
		}
	}else if (StringMatch(sk, VAR_SLOPE)) {
		m_Slope = data;
	}else if (StringMatch(sk, VAR_CHWIDTH)) {
		m_chWidth = data;
	}else if (StringMatch(sk, REACH_DEPTH)) {
		 m_chDepth = data;
	}else
	{
		throw ModelException("CASC2D_OF", "Set1DData", "parameter " + string(key) + " is not exist");
	}
}

void CASC2D_OF::Set2DData(const char* key, int n, int col, float** data) {
	if (StringMatch(key, Tag_FLOWIN_INDEX_D8)) {
		m_flowInIndex = data;
	}
}

void CASC2D_OF::SetReaches(clsReaches* rches) {
	// �ӵ����
	if (nullptr == rches) {
		throw ModelException(MID_MUSK_CH, "SetReaches", "The reaches input can not to be NULL.");
	}
	m_nreach = rches->GetReachNumber();
	m_reachLayers = rches->GetReachLayers();
	m_downStreamReachId = rches->GetDownStreamID();
	if (nullptr == m_chSinuosity) rches->GetReachesSingleProperty(REACH_SINUOSITY, &m_chSinuosity);

}

void CASC2D_OF::SetSubbasins(clsSubbasins* subbsns) {
	if (m_subbasinsInfo == nullptr) {
		m_subbasinsInfo = subbsns;
		// m_nSubbasins = m_subbasinsInfo->GetSubbasinNumber(); // Set in SetValue()! lj
		m_subbasinIDs = m_subbasinsInfo->GetSubbasinIDs();
	}
}

void CASC2D_OF::SetRasterPositionDataPointer(const char* key, int** positions) {
	m_RasterPostion = positions;
}
void CASC2D_OF::SetScenario(Scenario* sce) {
}


void CASC2D_OF::Get1DData(const char* key, int* n, float** data) {
	string sk(key);
	*n = m_nCells;
	if (StringMatch(sk, VAR_QOVERLAND)) {
		*data = m_chQ;
	}else if (StringMatch(sk, VAR_SUR_WRT_DEPTH)) {
		*data = m_surWtrDepth;
	}else if (StringMatch(sk, VAR_CH_WRT_DEPTH)){
		*data = m_chWtrDepth;
	}
}

void CASC2D_OF::Get2DData(const char* key, int* n, int* col, float*** data) {	//const char* key��keyָ������ݲ��ܸı�
}


void CASC2D_OF::GetValue(const char* key, float* value) {
	string sk(key);
	// ��ˮ������
	if (StringMatch(sk, VAR_OUTLET_Q)) {
		*value = m_outQ;
	}
	// ��ˮ������
	else if (StringMatch(sk, VAR_OUTLET_V)) {
		*value = m_outV;
	}
}

TimeStepType CASC2D_OF::GetTimeStepType() {
	return TIMESTEP_HILLSLOPE;
}

//void CASC2D_OF::SetReachDepthData( FloatRaster* positions) {
//	m_reachDepth = positions;
//}

bool CASC2D_OF::CheckInputData() {

	// �ӵ��ڵĺӵ�ˮ���ʼ��Ϊ�ӵ��ĳ�ʼˮ��
	//if (m_InitialInputs)
	//{
	//	for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
	//		int nReaches = it->second.size();
	//		for (int i = 0; i < nReaches; ++i) {
	//			int reachIndex = it->second[i]; // index in the array, from 0
	//			vector<int> &curReachCells = m_reachs[reachIndex];
	//			int n = curReachCells.size();
	//			for (int iCell = 0; iCell < n; ++iCell) {
	//				int cellIndex = curReachCells[iCell];
	//				m_chWtrDepth[cellIndex] = m_surSdep[cellIndex];
	//			}
	//		}
	//	}
	//	m_InitialInputs = false;
	//}
    return true;
}


void CASC2D_OF::InitialOutputs() {
	if (nullptr == m_chQ) Initialize1DArray(m_nCells, m_chQ, 0.f);
	if (nullptr == m_surSdep) Initialize1DArray(m_nCells, m_surSdep, 0.f);
	// ��ʼ���ӵ�ˮ��,��ʵ���ｫ�ӵ�����ĳ�ʼˮ���ʱ��Ϊ0
	// �ӵ���ĺӵ���ʼˮ����Ϊ0
	if (nullptr == m_chWtrDepth) Initialize1DArray(m_nCells, m_chWtrDepth, 0.f);
	//m_outQ = 0.0;
	//m_outV = 0.0;
	if (m_InitialInputs) {
		// find source cells the reaches
		m_sourceCellIds = new int[m_nreach];
		//m_qsInput = new float[m_chNumber+1];

		for (int i = 0; i < m_nreach; ++i) {
			m_sourceCellIds[i] = -1;
			//m_qsInput[i] = 0.f;
		}
		int reachIndex = 0;
		for (int i = 0; i < m_nCells; i++) {
			if (FloatEqual(m_streamLink[i], NODATA_VALUE)) {
				continue;
			}
			// ��ǰ�ӵ���id
			int reachId = (int)m_streamLink[i];
			bool isSource = true;
			// �жϵ�ǰդ��Ԫ��ÿ��������Ԫ���Ƿ�͵�ǰդ������ͬһ���ӵ�
			// ����ǣ���ǰդ��Ԫ���Ǹúӵ���Դͷ
			for (int k = 1; k <= (int)m_flowInIndex[i][0]; ++k) {
				int flowInId = (int)m_flowInIndex[i][k];
				int flowInReachId = (int)m_streamLink[flowInId];
				if (flowInReachId == reachId) {
					isSource = false;
					break;
				}
			}
			// �����ǰդ��Ԫû��������Ԫ����ǰդ��Ԫ�Ǹúӵ���Դͷ
			if ((int)m_flowInIndex[i][0] == 0) {
				isSource = true;
			}
			// reachId�Ǻӵ���ʵ��id�����ܲ��Ǵ�0��ʼ�ģ�Ҳ��֪���м��ǲ����жϿ������������
			// reachIndex�Ǹ��ӵ���0��ʼ������ţ���������ӵ�
			// ����ǰդ��Ԫ��Ϊ��ǰ�ӵ���Դͷ
			if (isSource) {
				// ����m_idToIndexû�г�ʼ������������Ӹ������߼�
				// ���m_idToIndex�в�������ǰ�ӵ���id,�Ͱ�reachId��reachIndex��ӳ���ϵ����m_idToIndex
				if (m_idToIndex.find(reachId) == m_idToIndex.end())
				{
					m_idToIndex.insert(pair<int, int>(reachId, reachIndex));
				}
				//int reachIndex = m_idToIndex[reachId];
				// ��reachIndex���ӵ���Դͷդ��Ԫid��i
				m_sourceCellIds[reachIndex] = i;
				reachIndex++;
			}
		}

		//for(int i = 0; i < m_chNumber; i++)
		//    cout << m_sourceCellIds[i] << endl;

		// get the cells in reaches according to flow direction
		for (int iCh = 0; iCh < m_nreach; iCh++) {
			// ��ǰ�ӵ���Դͷ
			int iCell = m_sourceCellIds[iCh];
			// �ӵ���id
			int reachId = (int)m_streamLink[iCell];
			// ���ŵ�ǰ�ӵ������Σ�������������������úӵ�������դ�����m_reachs[iCh]
			while ((int)m_streamLink[iCell] == reachId) {
				m_reachs[iCh].push_back(iCell);
				iCell = (int)m_flowOutIndex[iCell];
			}
		}
		m_InitialInputs = false;

		//m_hCh = new float *[m_nreach];
		//m_qCh = new float *[m_nreach];

		////m_flowLen = new float *[m_chNumber];

		//m_qSubbasin = new float[m_chNumber];
		//for (int i = 0; i < m_chNumber; ++i) {
		//	int n = m_reachs[i].size();
		//	m_hCh[i] = new float[n];
		//	m_qCh[i] = new float[n];

		//	//m_flowLen[i] = new float[n];

		//	m_qSubbasin[i] = 0.f;

		//	//int id;
		//	//float dx;
		//	for (int j = 0; j < n; ++j) {
		//		m_hCh[i][j] = 0.f;
		//		m_qCh[i][j] = 0.f;
		//	}
		//}

	}
}


int CASC2D_OF::Execute() {
	CheckInputData();
	InitialOutputs();
	/* Overland and channel depth updating												*/
	double sub_t1 = TimeCounting();
	OvrlDepth();
	double sub_t2 = TimeCounting();
	//cout << "overland depth end, cost time: " << sub_t2 - sub_t1 << endl;

	ChannDepth();
	double sub_t3 = TimeCounting();
	//cout << "channel depth  end, cost time: " << sub_t3 - sub_t2 << endl;

	/* Overland and channel flow routing												*/
	OvrlRout();
	double sub_t4 = TimeCounting();
	//cout << "overland route  end, cost time: " << sub_t4 - sub_t3 << endl;

	ChannRout();
	double sub_t5 = TimeCounting();
	//cout << "channel route  end, cost time: " << sub_t5 - sub_t4 << endl;

	/* Flow routing at the outlet																*/
	RoutOutlet();
	double sub_t6 = TimeCounting();
	//cout << "route outlet  end, cost time: " << sub_t6 - sub_t5 << endl;
	cout << "timestamp  end, cost time: " << sub_t6 - sub_t1 << endl;
	return 0;
}






/*************************���µر������*******************************/
void CASC2D_OF::OvrlDepth()
{
	
	float hov;

	/**********************************************/
	/*    Updating overland depth (water balance) */
	/**********************************************/

	/* Applying the Rainfall to each Grid Cell within the Watershed */
	// ���������ڵ�ÿ��դ��Ԫ
	for (int i = 0; i < m_nCells; i++) {
		/* դ��Ԫ�ϸ�ʱ�䲽�����仯�ģ��ĵر�ˮ��� = �ϸ�ʱ�䲽����ˮ������ * ʱ�䵥Ԫ / դ�����	*/
		hov = m_chQ[i] * m_dt / (m_cellWth*m_cellWth);
		/*�߼������h�ĸ���ɾȥ���ꡢ������������Ӱ�죬��Ϊ����ģ���Ѿ����������������Ǿ����ٶ������ˮ��仯*/
		//if (i % 100 == 0)
		//{
		//	cout << "m_chQ: " << m_chQ[i] << " m_surWtrDepth: " << m_surWtrDepth[i] << " hov: " << hov << " hov-new: " << hov + m_surWtrDepth[i] << endl;
		//}
		hov = hov + m_surWtrDepth[i] / 1000.f; //  mm -> m
		/* ����ȫ�ֵر�ˮ��*/
		m_surWtrDepth[i] = hov * 1000.f;			// m -> mm
		/* ����ǰʱ�䲽���ڵĵر����ٱ仯��Ϊ0*/
		m_chQ[i] = 0.0;

	}
	
}

/*************************�������*******************************/
void CASC2D_OF::OvrlRout()
{

	int j, k, jj, kk, l;
	/* ����դ��Ԫ*/
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		//cout << "iCell " << iCell << endl;
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		int rightRow = curRow;
		int rightCol = curCol + 1;
		int rightCell = -1;
		bool isRightExists = false;
		int belowRow = curRow + 1;
		int belowCol = curCol;
		int belowCell = -1;
		bool isBelowExists = false;
		//double sub_t1 = TimeCounting();
		// ����ҷ���դ�����±�һ����iCell+1
		if (iCell + 1 < m_nCells && iCell < m_RasterPostion[iCell + 1][0] == rightRow && m_RasterPostion[iCell + 1][1] == rightCol) {
			isRightExists = true;
			rightCell = iCell + 1;
		}
		// ����·���դ�����±�һ��С�ڵ���iCell+������ֻ��Ҫ�ӵ�ǰդ��Ԫ����һ��դ��Ԫ����m_ncols����Ԫ��Χ�ڲ��Ҽ���
		for (int d_col = 1; d_col <= m_ncols; d_col++) {
			// �ҵ����һ��դ��Ͳ�����
			if (iCell + d_col >= m_nCells)
			{
				break;
			}
			// �ҵ��˾Ͳ�����
			if (m_RasterPostion[iCell + d_col][0] == belowRow && m_RasterPostion[iCell + d_col][1] == belowCol) {
				isBelowExists = true;
				belowCell = iCell + d_col;
				break;
			}
		}
		//double sub_t2 = TimeCounting();
		//cout << "find right and below costs: " << sub_t2 - sub_t1 << endl;
		// ����ҷ�դ��Ԫ��Ϊ�գ������ҷ�դ��Ԫ
		if (isRightExists)
		{
			ovrl(iCell, rightCell);
		}
		// ����·�դ��Ԫ��Ϊ�գ������·�դ��Ԫ
		if (isBelowExists)
		{
			ovrl(iCell, belowCell);
		}
		//double sub_t3 = TimeCounting();
		//cout << "ovrl costs: " << sub_t3 - sub_t2 << endl;
	}
}

void CASC2D_OF::ovrl(int icell, int rbCell)
{
	double sub_t1 = TimeCounting();
	int jfrom, kfrom, jto, kto;

	float a = 1.0;

	float vel = 0.0;

	float so, sf, dhdx, hh, rman, alfa, dqq, stordepth;

	so = (m_dem[icell] - m_dem[rbCell]) / m_cellWth;			/* �Ӵ��¶�*/

	dhdx = (m_surWtrDepth[rbCell] - m_surWtrDepth[icell]) / 1000.0f / m_cellWth;		/* dh/dx, �ر�ˮ��/դ��Ԫ��ȣ����ر�ˮ���¶�*/

	sf = so - dhdx + (float)(1e-30);	/* Ħ���¶�*/

	hh = m_surWtrDepth[icell] /1000.0f;                  /* �ر���ˮ��*/

	rman = m_ManningN[icell];		 /* ����ϵ��*/

	/* �ںӵ���*/
	//if (chancheck == 1 && link[j][k] > 0)
	if (!FloatEqual(m_streamLink[icell], NODATA_VALUE))
	{
		/* ���դ��Ԫ�ϵ���ˮ��� > �ӵ���ȣ���ӵ��Ϸ��ĵر�ˮ��=դ��Ԫ�ϵ�����ˮ���-�ӵ���ȣ�����ر�ˮ��=0.0*/
		if (m_surSdep[icell] > m_chDepth[icell])
		{
			/* �ر�ˮ�� = �ӵ���դ��Ԫ�ϵ�����ˮ��� - �ӵ���ȣ����߳��ӵ����ֵ���ȣ�*/
			stordepth = (m_surSdep[icell] - m_chDepth[icell]) / 1000.0f;
			//cout << "stordepth >: " << stordepth << " m_surSdep: " << m_surSdep[icell] << " m_chDepth: " << m_chDepth[icell] << endl;
		}
		else
		{
			stordepth = 0.0f;
			//cout << "stordepth zero: " << stordepth << " m_surSdep: " << m_surSdep[icell] << " m_chDepth: " << m_chDepth[icell] << endl;

		}
	}
	/* �ںӵ��⣬�ر�ˮ�� = դ��Ԫ�ϵ���ˮ���*/
	else
	{
		stordepth = m_surSdep[icell] / 1000.0f;
		//cout << "stordepth out: " << stordepth << " m_surSdep: " << m_surSdep[icell] << " m_chDepth: " << m_chDepth[icell] << endl;

	}

	/* Ħ���Ƚ� < 0��˵������դ��Ԫ֮���ˮ���ٶȷ����ı䣬���Ҫ�����¸�դ��Ԫ�ĵر�ˮ��*/
	if (sf < 0)
	{
		/* ��һ���ر�Ԫ�ĵر���ˮ��*/
		hh = m_surWtrDepth[rbCell] / 1000.0f;		// mm -> m

		if (hh <= 0.000001)
		{
			hh = 0.0f;
		}
		/* ��һ���ر�Ԫ������ϵ��*/
		rman = m_ManningN[rbCell];
		/* ��һ���ر�Ԫ�ںӵ���*/
		if (!FloatEqual(m_streamLink[rbCell], NODATA_VALUE))
			//if (chancheck == 1 && link[jj][kk] > 0)
		{
			/* ��һ��դ��Ԫ�ϵ���ˮ��� > �ӵ���ȣ����Ϊ��һ���ӵ��ر�Ԫ�ĺӵ����Ѿ�����ˮ*/
			if (m_surSdep[rbCell] > m_chDepth[rbCell])
			{
				/* ��һ���ر�Ԫ�ĵر�ˮ��*/
				stordepth =
					(m_surSdep[rbCell] - m_chDepth[rbCell]) / 1000.0f;
			}
			else
			{
				stordepth = 0.0f;
			}
		}
		else
		{
			stordepth = m_surSdep[rbCell] / 1000.0f;
		}
		//cout << "stordepth sf: " << stordepth << " m_surSdep: " << m_surSdep[icell] << " m_chDepth: " << m_chDepth[icell] << endl;

	}
	if (stordepth <= 0.000001)
	{
		stordepth = 0.0f;
	}
	/* ������ȴ��ڵر�ˮ��*/
	// todo ������京�壿
	if (hh >= stordepth)
	{
		/* alfa�Ǹ��������������µ�Ħ��Ƚ����������̬����*/
		alfa = (float)((pow(fabs(sf), 0.5)) / rman);

		/*	Note : The variable "a" represents the sign of the				*/
		/*	Friction Slope (Sf)	Computing Overland Flow								*/

		if (sf >= 0) a = 1.0;

		if (sf < 0) a = -1.0;
		/* dqq ʱ�䲽���ڵر������ʵı仯�� = alfa * h��(5/3)�η�*/
		dqq = (float)(a*m_cellWth*alfa*pow((hh - stordepth), 1.667));

		int i = icell;
		//if (i % 100 == 0)
		//{
		//	cout << "icell: " << icell << " rbCell: " << rbCell << " m_chQ: " << m_chQ[i] << " m_chQ-new: " << m_chQ[i] - dqq << " dqq: " << dqq << " hh: " << hh << " stordepth: " << stordepth << " alfa: " << alfa << endl;
		//}
		/* ���դ��Ԫ��ʱ�䲽���ڵĵر�������*/
		m_chQ[icell] = m_chQ[icell] - dqq;

		m_chQ[rbCell] = m_chQ[rbCell] + dqq;

		//cout << "m_chQ[icell]: " << m_chQ[icell] << " m_chQ[rbCell]: " << m_chQ[rbCell]  << endl;


	}	/* End of HH >= STORDEPTH */

}   /* End of OVRL */

/*************************���ºӵ��������*******************************/
void CASC2D_OF::ChannDepth()
{
	int ic, j, l, k, jj;
	float wch, dch, sfactor, sdep_ov, inflowVol, vol_ov_in;
	for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
		int nReaches = it->second.size();
		// ������ǰͼ������кӵ�
		for (int i = 0; i < nReaches; ++i) {
			int reachId = it->second[i];
			// ����reachId���Һӵ���index��index�Ǻӵ��������е��±꣬��0��ʼ
			map<int, int>::iterator iter = m_idToIndex.find(reachId);
			if (iter != m_idToIndex.end()) {
				int reachIndex = iter->second;
				//int reachIndex = m_idToIndex.find(reachId);
				vector<int> &vecCells = m_reachs[reachIndex];
				int n = vecCells.size();
				for (int iCell = 0; iCell < n; ++iCell) {
					int idCell = vecCells[iCell];
					/* �ӵ���ȡ���ȡ�������*/
					wch = m_chWidth[idCell];
					dch = m_chDepth[idCell];
					//sfactor = m_chSinuosity[idCell];
					sfactor = 1;
					/* Find new channel depth after adding inflow volume				*/
					/* ��ǰʱ�䲽���ĺӵ�ˮ�����*/
					inflowVol = m_chQ[idCell] * m_dt;
					/* ... and the volume coming from the overland	(vol_ov_in) */
					/* �����ӵ���ȵ�ˮ�����*/
					// ����о��߼���ͨ��Ϊʲô�����ó�ʼˮ������ȣ�
					if (m_surSdep[idCell] / 1000.0f > dch)
						sdep_ov = m_surSdep[idCell] / 1000.0f - dch;
					else
						sdep_ov = 0.0;
					/* ��ǰʱ�䲽�����Եر�Ԫ��ˮ�����*/
					vol_ov_in = 0;

					if (m_surWtrDepth[idCell] / 1000.0f > sdep_ov)
					{
						vol_ov_in = (m_surWtrDepth[idCell] / 1000.0f - sdep_ov)*m_cellWth*m_cellWth;
						m_surWtrDepth[idCell] = sdep_ov * 1000.0f;
					}

					m_chWtrDepth[idCell] = newChnDepth(wch, dch, sfactor, idCell,
						(inflowVol + vol_ov_in));   // m

					/* Negative Depth in the Channel --> EXIT program					*/

					if (m_chWtrDepth[idCell] < 0.0)
					{
						if (m_chWtrDepth[idCell] < 0.0 && m_chWtrDepth[idCell] > -0.001)
							m_chWtrDepth[idCell] = 0.0;
					}

					m_chQ[idCell] = 0.0;
				}
			}
		
		}
	}
	
}

float CASC2D_OF::newChnDepth(float wch, float dch, float sfactor,int idCell, float addedVolume)
{
	float area_ch, vol_ch, area_init, vol_init,
		vol_final, newdepth;

	/* Channel area and volume																			*/

	area_ch = wch * dch;
	vol_ch = area_ch * m_cellWth * sfactor;

	/* Calculates initial area and volume														*/

	if (m_chWtrDepth[idCell] <= dch)
		area_init = wch * m_chWtrDepth[idCell];
	else
		area_init = (m_chWtrDepth[idCell] - dch) * m_cellWth + area_ch;

	vol_init = area_init * m_cellWth * sfactor;

	/* After adding new volume calculates volume										*/

	vol_final = vol_init + addedVolume;

	/* ... and depth corresponding to the final volume							*/

	if (vol_final > vol_ch)
		newdepth = dch + (vol_final - vol_ch) / (m_cellWth*m_cellWth*sfactor);
	else
		newdepth = vol_final / (wch*m_cellWth*sfactor);

	return(newdepth);

}


/*************************�ӵ�����*******************************/
void CASC2D_OF::ChannRout()
{
	for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
		int nReaches = it->second.size();
		for (int i = 0; i < nReaches; ++i) {
			int reachId = it->second[i]; // index in the array, from 0
			int reachIndex = 0;
			map<int, int>::iterator iter = m_idToIndex.find(reachId);
			if (iter != m_idToIndex.end()) {
				reachIndex = iter->second;
				vector<int> &curReachCells = m_reachs[reachIndex];
				int n = curReachCells.size();
				for (int iCell = 0; iCell < n; ++iCell) {
					int cellIndex = curReachCells[iCell];
					chnchn(reachIndex, reachId, nReaches, iCell, curReachCells);
				}
			}

		}
	}
}
/*
curReachIndex: �ӵ����
iCell: �ӵ��еĵ�iCell���ڵ�
curCellIndex: �ӵ��еĵ�iCell���ڵ���դ�������е��±�
*/
void CASC2D_OF::chnchn(int curReachIndex,int curReachId,int nReaches,int iCell, vector<int> curReachCells)
{

	float a = 1.0;
	float vel = 0.0;
	float wch, dch, sslope, rmanch, sfactor, so, sf, hh, dhdx,dq, stordep, hchan, dq_ov;

	int  j, k, jj, kk, jjj, iic, ijun, ill;
	int curCellIndex = curReachCells[iCell];


	/* Note : JJJ is a check to see when the end of									*/
	/*        a channel link has been reached.											*/

	//int nextNextCell = m_reachs[curReachIndex][iCell + 2];
	//int nextReachId = m_downStreamReachId.
	/* Channel characteristics :																		*/

	wch = m_chWidth[curCellIndex];				/* width						*/
	dch = m_chDepth[curCellIndex];					/* depth						*/
	//dch = m_reachDepth->GetRasterDataPointer()[curCellIndex];				
	sslope = m_Slope[curCellIndex];					/* side slope					*/
	rmanch = m_ManningN[curCellIndex];		/* manning's n				*/
	//sfactor = m_chSinuosity[curReachIndex];	/* sinuosity factor,use the reach's sinuosity instead of that on every cell of reach temporary	*/
	sfactor = 1;
	stordep = m_surSdep[curCellIndex];			/* Storage depth			*/
	hchan = m_chWtrDepth[curCellIndex];		/* Channel water depth*/
	hh = m_chWtrDepth[curCellIndex] - stordep;

	/* Channel slope																								*/
	/* row and column of following node in link IC									*/
	/* index of following node in link curReachIndex*/
	// ��ǰ�ӵ���ǰ�ڵ����һ���ڵ�ĵ��
	int nextCellIndex = -1;
	// �����ǰ�ڵ㲻�ǵ�ǰ�ӵ������һ���ڵ�
	if (iCell < curReachCells.size() - 1)
	{
		nextCellIndex = curReachCells[iCell + 1];
		so = (m_dem[curCellIndex] - m_dem[nextCellIndex]) / (m_cellWth*sfactor);
	}
	// �����ǰ�ڵ��ǵ�ǰ�ӵ������һ���ڵ�
	else
	{
		// �ҵ����κӵ��ĵ�һ���ڵ�
		int nextReachID = m_downStreamReachId[curReachId];
		int nextReachIndex = 0;
		map<int, int>::iterator iter = m_idToIndex.find(nextReachID);
		// �����ǰ�ӵ����������εĺӵ����������κӵ���
		if (iter != m_idToIndex.end()) {
			nextReachIndex = iter->second;
			vector<int> nextReachCells = m_reachs[nextReachIndex];
			int nextReachFistCellIndex = nextReachCells[0];
			nextCellIndex = nextReachFistCellIndex;
			so = (m_dem[curCellIndex] - dch - m_dem[nextCellIndex] + m_chDepth[nextCellIndex]) / (m_cellWth*sfactor);
			//ijun = nextReachIndex;
		}
		// �����ǰ�ӵ��������εĺӵ�
		else
		{
			nextCellIndex = -1;
			so = 0.f;
		}
	}
	// nextCellIndex = -1��ζ�����һ���ӵ������һ���ڵ㣬����ڵ��ˮ������������
	if (nextCellIndex > -1)
	{
		/* hch[j][k]դ��Ԫ�ϵĺӵ�ˮ�dhdx��ˮ���¶�*/
		//dhdx = (m_reachDepth->GetRasterDataPointer()[nextCellIndex] - m_reachDepth->GetRasterDataPointer()[curCellIndex] / (m_cellWth*sfactor));
		dhdx = (m_chWtrDepth[nextCellIndex] - m_chWtrDepth[curCellIndex]) / (m_cellWth*sfactor);

		/* Ħ���¶�*/
		sf = (float)(so - dhdx + 1e-30);

		/* Nota de Jorge: Sf se deberia quedar con el mismo signo */
		/* sf Ӧ�ñ�����ͬ�ķ��ţ���sf����ֵ�����С������sfΪ��ֵ*/
		if (fabs(sf) < 1e-20) sf = (float)(1e-20);

		if (sf < 0.0)
		{
			a = (float)(-1.0*a);

			/* �����ǰ�ڵ��ǵ�ǰ�ӵ������һ���ڵ㣬���ȡ���κӵ��ĵ�һ���ڵ������*/
			if (iCell >= curReachCells.size() - 1)
			{
				/* Take channel chars. of the 1st node of downstream link */
				// ��֤����������е����κӵ�id�Ƿ���ͬ����������arcgis����֤
				int nextReachID = m_downStreamReachId[curReachIndex];
				map<int, int>::iterator iter = m_idToIndex.find(nextReachID);
				if (iter != m_idToIndex.end()) {
					int nextReachIndex = iter->second;
					vector<int> nextReachCells = m_reachs[nextReachIndex];
					int nextReachFistCellIndex = nextReachCells[0];
					wch = m_chWidth[nextReachFistCellIndex];
					dch = m_chDepth[nextReachFistCellIndex];
					sslope = m_Slope[nextReachFistCellIndex];
					rmanch = m_ManningN[nextReachFistCellIndex];
					//sfactor = m_chSinuosity[nextReachID];
					sfactor = 1;
				}

			}
			/*�����ǰ�ڵ㲻�ǵ�ǰ�ӵ������һ���ڵ㣬���ȡ��ǰ�ӵ�����һ���ڵ�*/
			else
			{
				/*Take channel chars. of the next node within current link */

				wch = m_chWidth[nextCellIndex];
				dch = m_chDepth[nextCellIndex];
				sslope = m_Slope[nextCellIndex];
				rmanch = m_ManningN[nextCellIndex];
				//sfactor = m_chSinuosity[curReachIndex];
				sfactor = 1;
			}

			/* hh = �ӵ�ˮ�� - ��ˮ��ȣ�hh < 0����ˮ��*/
			stordep = m_surSdep[nextCellIndex];
			hchan = m_chWtrDepth[nextCellIndex];
			hh = m_chWtrDepth[nextCellIndex] - stordep;

		}

		/* Determining discharge 																				*/
		/* ����ӵ�������*/
		dq = chnDischarge(hchan, hh, wch, dch, stordep, rmanch, a, sf, sfactor);

		/* Transfer flow from cell (j,k) to (jj,kk)											*/
		/* �ӵ��������� ������/s*/
		m_chQ[curCellIndex] = m_chQ[curCellIndex] - dq;
		m_chQ[nextCellIndex] = m_chQ[nextCellIndex] + dq;
	}

}		/* End of CHNCHN */


/*************************�ӵ���ˮ*******************************/
void CASC2D_OF::RoutOutlet()
{
	int ill;
	float hout, alfa, qoutch;

	qoutov = 0.0;
	qoutch = 0.0;
	/* FIRST:calculate the flow going out from the overl. portion		*/
	/* sovout��ˮ������դ��Ԫ�ϵ��¶�*/
	alfa = (float)(sqrt(sovout) / m_ManningN[m_idOutlet]);

	/* Discharge from overland flow.  NOTE: because the water from  */
	/* this part of the outlet overland cell was already "poured"		*/
	/* into the channel when updating the channel depth (channDepth)*/
	/* qoutov = 0 when the channel routing is selected							*/

	if (m_surWtrDepth[m_idOutlet] > m_surSdep[m_idOutlet])
	{
		qoutov =
			(float)(m_cellWth*alfa*pow((m_surWtrDepth[m_idOutlet] - m_surSdep[m_idOutlet]), 1.667));
	}

	/* Overland water depth at outlet cell is reduced after taking	*/
	/* the outflow out of the cell														*/

	m_surWtrDepth[m_idOutlet] = (float)(m_surWtrDepth[m_idOutlet] - qoutov * m_dt / (pow(m_cellWth, 2.0)));

	/* SECOND:calculate the flow going out from the channel portion	*/
	//if (chancheck == 1 && hch[jout][kout] > sdep[jout][kout])
	if (m_chWtrDepth[m_idOutlet] > m_surSdep[m_idOutlet])
	{
		hout = m_chWtrDepth[m_idOutlet] - m_surSdep[m_idOutlet];

		qoutch = chnDischarge(m_chWtrDepth[m_idOutlet], hout, m_chWidth[m_idOutlet], m_chDepth[m_idOutlet],
			m_surSdep[m_idOutlet], m_ManningN[m_idOutlet], 1, m_Slope[m_idOutlet], m_chSinuosity[m_idOutlet]);

		m_chQ[m_idOutlet] = m_chQ[m_idOutlet] - qoutch;
	}

	/* The total outflow at the basin's outlet is given by adding		*/
	/* the outflow from the overland & channel portion of the cell	*/

	m_outQ = qoutov + qoutch;

	/* Keeping Track of the Total Outflow Volume										*/

	m_outV = m_outV + m_outQ * m_dt;

	/* Checking to see if the Peak Flow has been reached						*/

	//if (qout > qpeak)
	//{
	//	qpeak = qout;
		//tpeak = (float)(iter*dt / 60.0);  /* ��¼��ֵʱ��*/
	//}

	/* Populating the Output Flows at the Watershed Outlet					*/
	/* �����û��Զ���ĳ�ˮ�ڵ���������ʱ����*/
	//for (ill = 1; ill <= ndis; ill++)
	//{
	//	if (jout == iq[ill][1] && kout == iq[ill][2])
	//	{
	//		q[ill] = qout;
	//	}
	//}
}

/* ����ӵ�������*/
float CASC2D_OF::chnDischarge(float hchan, float hh, float wch, float dch, float stordep, float rmanch, float a, float sf, float sfactor)
{
	float area, wp, dQ, vol_ch_avail;

	/* Calculates flow area and wetted perimeter										*/

	if (hchan <= dch)
	{
		area = wch * hh;
		wp = (float)(wch + 2 * hh);
	}
	else
	{
		area = wch * (dch - stordep) + m_cellWth * (hchan - dch);
		wp = (float)(wch + 2 * (dch - stordep) + 2 * (m_cellWth - wch) + 2 * (hchan - dch));
	}

	dQ = (float)(a*(sqrt(fabs(sf)) / rmanch)*
		(pow(area, 1.6667)) / (pow(wp, 0.6667)));

	/* Limit the outflow by availability														*/
	/* ������������*/
	vol_ch_avail = area * m_cellWth * sfactor;

	if (dQ*m_dt > vol_ch_avail) dQ = vol_ch_avail / m_dt;

	return(dQ);

}



