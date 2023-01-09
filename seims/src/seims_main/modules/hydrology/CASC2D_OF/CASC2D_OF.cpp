#include "CASC2D_OF.h"
#include "text.h"
using namespace std;

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

bool CASC2D_OF::CheckInputData() {

    return true;
}


void CASC2D_OF::InitialOutputs() {
	if (nullptr == m_chQ) Initialize1DArray(m_nCells, m_chQ, 0.f);
	// m_surSdep(m)
	if (nullptr == m_surSdep) Initialize1DArray(m_nCells, m_surSdep, 0.f);
	// ��ʼ���ӵ�ˮ��,��ʵ���ｫ�ӵ�����ĳ�ʼˮ���ʱ��Ϊ0
	// �ӵ���ĺӵ���ʼˮ����Ϊ0(m)
	if (nullptr == m_chWtrDepth) Initialize1DArray(m_nCells, m_chWtrDepth, 0.f);
	if (m_InitialInputs) {
		output_icell = 50676;
		output_icell_min = 835;
		output_icell_max = 860;
		counter = 0;
		// find source cells the reaches
		m_sourceCellIds = new int[m_nreach];

		for (int i = 0; i < m_nreach; ++i) {
			m_sourceCellIds[i] = -1;
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

		// ��ʼ��ÿ��դ���Ӧ���ҡ��·�դ��
		for (int iCell = 0; iCell < m_nCells; iCell++) {
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
			vector<int> rbCells(2);
			// ����ҷ���դ�����±�һ����iCell+1
			if (iCell + 1 < m_nCells && rightCol < m_ncols &&  m_RasterPostion[iCell + 1][0] == rightRow && m_RasterPostion[iCell + 1][1] == rightCol) {
				isRightExists = true;
				rightCell = iCell + 1;
			}
			rbCells[0] = rightCell;

			// ����·���դ�����±�һ��С�ڵ���iCell+������ֻ��Ҫ�ӵ�ǰդ��Ԫ����һ��դ��Ԫ����m_ncols����Ԫ��Χ�ڲ��Ҽ���
			for (int d_col = 1; d_col <= m_ncols; d_col++) {
				// �ҵ����һ��դ��Ͳ�����
				if (iCell + d_col >= m_nCells)
				{
					break;
				}
				// �ҵ��˾Ͳ�����
				if (m_RasterPostion[iCell + d_col][0] == belowRow && m_RasterPostion[iCell + d_col][1] == belowCol) {
					belowCell = iCell + d_col;
					break;
				}
			}
			rbCells[1] = belowCell;
			m_rbcellsMap.insert(pair<int, vector<int>>(iCell, rbCells));
		}
		m_InitialInputs = false;

	}
}


int CASC2D_OF::Execute() {
	InitialOutputs();
	# ifdef IS_DEBUG
	std::ostringstream oss;
	//oss << "F:\\program\\lisflood\\test\\Summ_file_" << counter << ".txt";
	//string Summ_file = oss.str();
	oss << "F:\\program\\\seims\\\data\\log\\flow_" << counter << ".txt";
	string Summ_file = oss.str();
	//string Summ_file = "F:\\program\\lisflood\\test\\position.txt";            // ���ÿ��դ��Ԫ������λ��

	//string Summ_file = "F:\\program\\lisflood\\test\\Summ_file.txt";       

	//�ļ�������ɾ��
	//if (_access(Summ_file.c_str(), 0) == 0) {
	//	if (remove(Summ_file.c_str()) == 0) {
	//		cout << "succeed to delete casc2d output file " << Summ_file.c_str() << endl;
	//	}
	//	else {
	//		cout << "failed to delete casc2d output file.  " << Summ_file.c_str() << endl;
	//	}
	//}
	//���λ������
	//if (counter == 0) {
	//	Summ_file_fptr.open(Summ_file.c_str(), std::ios::out | std::ios::app);
	//	OutputPosition();
	//}
	// ָ����������֮��ʼ���
	//if ((counter >= output_icell_min && counter <= output_icell_max) && !Summ_file_fptr.is_open()) {
	//	Summ_file_fptr.open(Summ_file.c_str(), std::ios::out | std::ios::app);
	//}
	// ��һ�ε����Ϳ�ʼ���
	if ( !Summ_file_fptr.is_open()) {
		Summ_file_fptr.open(Summ_file.c_str(), std::ios::out | std::ios::app);
	}

	if (counter == 0)
	{
		buildPositionIndex();
	}
	# endif
	counter++;

	double sub_t1 = TimeCounting();
	OvrlDepth();
	ChannDepth();
	OvrlRout();
	ChannRout();
	RoutOutlet();
	//printFlow();
	//if (counter >= output_icell_min && counter <= output_icell_max)
	//{
	//	printFlow();
	//}
	double sub_t2 = TimeCounting();
	cout << "casc2d_sed  end, cost time: " << sub_t2 - sub_t1 << endl;
	if (Summ_file_fptr.is_open()) {
		Summ_file_fptr.close();
	}
	return 0;
}

// ������ж�Ӧ��iCell��Ϣ
void CASC2D_OF::OutputPosition() {
	int last_row = -1;
	int start_col = 0;
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		if (iCell == 0)
		{
			for (int i = 1; i <= m_ncols; i++)
			{
				Summ_file_fptr << std::left << setw(7) << setfill(' ') << i;
			}
		}
		// ����
		if (last_row != curRow)
		{
			Summ_file_fptr << endl;
			// �������
			for (int i = 1; i < curCol; i++)
			{
				Summ_file_fptr << setfill(' ') << setw(7) << ' ';
			}
		}
		Summ_file_fptr << std::left << setw(7) << setfill(' ') << iCell;
		last_row = curRow;
	}
}

void CASC2D_OF::printFlow() {
	int last_row = -1;
	int start_col = 0;
	int cols = 0;
	int lastrow_firstcol = 0;
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		// ��ӡ�к�
		if (iCell == 0)
		{
			for (int i = 1; i <= m_ncols; i++)
			{
				Summ_file_fptr << std::left << setfill(' ') << setw(26) << i;
			}
		}
		// ���д�ӡ���з�������
		if (last_row != curRow)
		{
			// �����Ժ��ӡy��������,dqq>0������,��һ��֮�ϲ��������
			if (last_row != -1)
			{
				Summ_file_fptr << endl;
				// �������
				for (int i = 1; i < lastrow_firstcol; i++)
				{
					Summ_file_fptr << setfill(' ') << setw(26) << ' ';
				}
				for (int i = 1; i <= cols; i++)
				{
					// todo: ������µ���������ȷ����
					if (m_RasterNeighbor[iCell][5] > 0)
					{
						Summ_file_fptr << "��" << std::left << setw(11) << setfill(' ') << m_RasterNeighbor[iCell][5];
					}
					else if(m_RasterNeighbor[iCell][5] == 0)
					{
						Summ_file_fptr << "|" << std::left << setw(11) << setfill(' ') << m_RasterNeighbor[iCell][5];
					}
					else {
						Summ_file_fptr << "��" << std::left << setw(11) << setfill(' ') << -m_RasterNeighbor[iCell][5];
					}
					Summ_file_fptr << setfill(' ') << setw(14) << ' ';
				}
				cols = 0;
			}

			Summ_file_fptr << endl;
			// �������
			for (int i = 1; i < curCol; i++)
			{
				Summ_file_fptr << setfill(' ') << setw(26) << ' ';
			}
			lastrow_firstcol = curCol;
		}
		// ��ӡλ��
		Summ_file_fptr << std::left << setw(7) << setfill(' ') << iCell;
		// ��ӡˮ��
		Summ_file_fptr << std::left << setw(7) << setfill(' ') << m_surWtrDepth[iCell];
		// ��ӡx��������,dqq>0������
		if (m_RasterNeighbor[iCell][4] > 0)
		{
			Summ_file_fptr << "��" << std::left << setw(11) << setfill(' ') << m_RasterNeighbor[iCell][4];
		}
		else if (m_RasterNeighbor[iCell][4] == 0) {
			Summ_file_fptr << "--" << std::left << setw(10) << setfill(' ') << m_RasterNeighbor[iCell][4];
		}
		else
		{
			Summ_file_fptr << "��" << std::left << setw(11) << setfill(' ') << -m_RasterNeighbor[iCell][4];
		}
		cols++;
		last_row = curRow;
	}
}

// ����λ��������dqq���飬�������� ��Ӧ 0123����dqq��Ӧ4����dqq��Ӧ5
void CASC2D_OF::buildPositionIndex() {
	// λ��
	m_RasterNeighbor = new int* [m_nCells];
	for (int i = 0; i < m_nCells; i++) {
		m_RasterNeighbor[i] = new int[6];
		for (int j = 0; j < 6; j++)
		{
			m_RasterNeighbor[i][j] = -1;
		}
	}
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		
		// �������λ��
		if (curRow != 0)
		{
			for (int i = iCell - 1; i >= iCell - m_ncols; i--)
			{
				// ��ֹ����Խ��
				if (i < 0)
				{
					break;
				}
				if ( m_RasterPostion[i][0] == (curRow - 1) && m_RasterPostion[i][1] == curCol)
				{
					m_RasterNeighbor[iCell][0] = i;
					break;
				}
			}
		}

		// �������λ��
		if (curRow != m_nrows - 1)
		{
			for (int i = iCell + 1; i <= iCell + m_ncols; i++)
			{
				// ��ֹ����Խ��
				if (i > m_nCells - 1)
				{
					break;
				}
				if (m_RasterPostion[i][0] == (curRow + 1) && m_RasterPostion[i][1] == curCol)
				{
					m_RasterNeighbor[iCell][1] = i;
					break;
				}
			}
		}

		// �������λ��(�������У���һ������һ��)
		if (iCell != 0 &&  curRow == m_RasterPostion[iCell - 1][0] && (curCol -1) == m_RasterPostion[iCell - 1][1])
		{
			m_RasterNeighbor[iCell][2] = iCell - 1;
		}
		// �������λ��(����ұ��У���һ������һ��)
		if (iCell != (m_nCells -1) && curRow == m_RasterPostion[iCell + 1][0] && (curCol + 1) == m_RasterPostion[iCell + 1][1])
		{
			m_RasterNeighbor[iCell][3] = iCell + 1;
		}

	}
}

void CASC2D_OF::traceSource(int icell) {
	while (hasSource(icell)) {

	}
}

bool CASC2D_OF::hasSource(int icell) {
	return true;
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
		/* dqov[j][k]�ǲ��� m3/s */
		/* hov ���� = ���� * ʱ�� / դ�����  m */
		hov = m_chQ[i] * m_dt / (m_cellWth*m_cellWth);
		/*�߼������h�ĸ���ɾȥ���ꡢ������������Ӱ�죬��Ϊ����ģ���Ѿ����������������Ǿ����ٶ������ˮ��仯*/
		hov = hov + m_surWtrDepth[i] / 1000.f; //  mm -> m
		if (hov < 0.0 && hov > -0.01)
		{
			hov = 0.0f;
		}
		/* ����ȫ�ֵر�ˮ��*/
		m_surWtrDepth[i] = hov * 1000.f;			// m -> mm
		/* ����ǰʱ�䲽���ڵĵر����ٱ仯��Ϊ0*/
		m_chQ[i] = 0.0;
		//# ifdef IS_DEBUG
		//if (i == output_icell && Summ_file_fptr.is_open()) {
		//	Summ_file_fptr << "icell: " << output_icell << " surWtrDepth: " << m_surWtrDepth[i] << " m_surWtrDepth: " <<  endl;
		//}
		//if (isnan(hov) || isnan(m_surWtrDepth[i]) || isnan(m_chQ[i]) || isinf(hov) || isinf(m_surWtrDepth[i]) || isinf(m_chQ[i])) {
		//	if (Summ_file_fptr.is_open()) {
		//		Summ_file_fptr << " i: " << i << " m_surWtrDepth[i]: " << m_surWtrDepth[i] << " m_chQ[i]: " << m_chQ[i] << endl;
		//	}
		//}
		//#endif // IS_DEBUG
	}
	
}

/*************************�������*******************************/
void CASC2D_OF::OvrlRout()
{

	int j, k, jj, kk, l;
	int lastRow = 0;
	/* ����դ��Ԫ*/
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		//# ifdef IS_DEBUG
		//if (iCell == 0)
		//{
		//	for (int i = 1; i <= m_ncols; i++)
		//	{
		//		Summ_file_fptr<< std::left << setw(20) << i;
		//	}
		//	Summ_file_fptr << endl;
		//}
		//if (iCell == 0)
		//{
		//	for (int i = 0; i < curCol; i++)
		//	{
		//		Summ_file_fptr << std::left << setw(20);
		//	}
		//}
		//if (lastRow != curRow)
		//{
		//	Summ_file_fptr << endl;
		//	for (int i = 0; i < curCol; i++)
		//	{
		//		Summ_file_fptr << std::left << setw(20);
		//	}
		//}
		//#endif // IS_DEBUG
		map<int,vector<int>>::iterator it;
		it = m_rbcellsMap.find(iCell);
		if (it != m_rbcellsMap.end())
		{
			int dqq = 0.0f;
			int rightCell;
			// ����ҷ�դ��Ԫ��Ϊ�գ������ҷ�դ��Ԫ
			if ((it->second)[0] != -1)
			{
				rightCell = (it->second)[0];
# ifdef IS_DEBUG
				//if (counter >= 838 && Summ_file_fptr.is_open()) {
				//	Summ_file_fptr << "[ I:" << iCell << std::left << setw(2) << "r:" << rightCell << " " << std::left << setw(8) << dqq << "]";
				//}
				//if (counter >= 600 && Summ_file_fptr.is_open()) {
				//	Summ_file_fptr << "ic: " << iCell << " dep: " << m_surWtrDepth[iCell]  << " rc: " << rightCell << " dep: " << m_surWtrDepth[rightCell];
				//}
#endif // IS_DEBUG
				dqq = ovrl(iCell, rightCell);
				m_RasterNeighbor[iCell][4] = dqq;
			}

			int belowCell;
			// ����·�դ��Ԫ��Ϊ�գ������·�դ��Ԫ
			if ((it->second)[1] != -1)
			{
				belowCell = (it->second)[1];
				# ifdef IS_DEBUG
				//if (counter >= 600 && Summ_file_fptr.is_open()) {
				//	Summ_file_fptr << "ic: " << iCell << " dep: " << m_surWtrDepth[iCell] << " bc: " << belowCell << " dep: " << m_surWtrDepth[belowCell] << " row: " << curRow << " col: " << curCol;
				//}
				#endif // IS_DEBUG
				dqq = ovrl(iCell, belowCell);
				m_RasterNeighbor[iCell][5] = dqq;
				//if (counter >= 838 && Summ_file_fptr.is_open()) {
				//	Summ_file_fptr << "[ I:"<< iCell << std::left << setw(2) <<  "b:" << belowCell << " " << std::left << setw(8)<< dqq << "]";
				//}

			}
		}
		# ifdef IS_DEBUG

		lastRow = curRow;
		#endif // IS_DEBUG


	}
}

float CASC2D_OF::ovrl(int icell, int rbCell)
{
	int jfrom, kfrom, jto, kto;

	float a = 1.0;

	float vel = 0.0;

	float so = 0.0f, sf = 0.0f, dhdx = 0.0f, hh = 0.0f, rman = 0.0f, alfa = 0.0f, dqq = 0.0f, stordepth = 0.0f;

	so = (m_dem[icell] - m_dem[rbCell]) / m_cellWth;			/* �Ӵ��¶�*/

	dhdx = (m_surWtrDepth[rbCell] - m_surWtrDepth[icell]) / 1000.0f / m_cellWth;		/* ˮ���¶�*/

	/* ��casc2d��sf����ΪĦ���Ƚ���ʵ�ʺ����Ǹ��ӱȽ���
	  * ���ӱȽ� = ����ˮ��Ƚ� - �ȶ���ˮ��Ƚ� �� ����ˮ��Ƚ� - �ӵ��¶ȱȽ�
	  * ���ӱȽ� < 0�������Ǻ飬�����ڲ�ǰ�����ӱȽ�Ϊ��
	  * ���ӱȽ� > 0��������飬�����ڲ��󣬸��ӱȽ�Ϊ��
	 */
	sf = so - dhdx + (float)(1e-30);	
	/* hh(m)���ر���ˮ��*/
	hh = m_surWtrDepth[icell] /1000.0f;                 
	/* ����ϵ��*/
	rman = m_ManningN[icell];		

	//if (isnan(dhdx) || isnan(m_surWtrDepth[rbCell]) || isnan(m_surWtrDepth[icell]) || isinf(dhdx) || isinf(m_surWtrDepth[rbCell]) || isinf(m_surWtrDepth[icell])) {
	//	if (Summ_file_fptr.is_open()) {
	//		Summ_file_fptr << " icell: " << icell << " m_surWtrDepth[icell]: " << m_surWtrDepth[icell] << " m_surWtrDepth[rbCell]: " << m_surWtrDepth[rbCell] << endl;
	//	}
	//}
	
	/* �ںӵ���*/
	if (!FloatEqual(m_streamLink[icell], NODATA_VALUE))
	{
		/* ���դ��Ԫ�ϵ���ˮ��� > �ӵ���ȣ���ӵ��Ϸ��ĵر�ˮ��=դ��Ԫ�ϵ�����ˮ���-�ӵ���ȣ�����ر�ˮ��=0.0*/
		if (m_surSdep[icell] > m_chDepth[icell])
		{
			/* �ȶ������ = �ӵ���դ��Ԫ�ϵ�����ˮ��� - �ӵ���ȣ����߳��ӵ����ֵ���ȣ�*/
			stordepth = m_surSdep[icell] - m_chDepth[icell] ;
		}
		else
		{
			stordepth = 0.0f;
		}
	}
	/* �ںӵ��⣬�ȶ������ = դ��Ԫ�ϵ���ˮ���*/
	else
	{
		stordepth = m_surSdep[icell];
	}

	/* ��casc2d��sf����ΪĦ���Ƚ���ʵ�ʺ����Ǹ��ӱȽ���
	  * ���ӱȽ� = ����ˮ��Ƚ� - �ȶ���ˮ��Ƚ� �� ����ˮ��Ƚ� - �ӵ��¶ȱȽ�
	  * ���ӱȽ� > 0�������Ǻ飬�����ڲ�ǰ�����ӱȽ�Ϊ��
	  * ���ӱȽ� < 0��������飬�����ڲ��󣬸��ӱȽ�Ϊ��
	  */
	if (sf < 0)
	{
		/* ��һ���ر�Ԫ�ĵر���ˮ��*/
		hh = m_surWtrDepth[rbCell] / 1000.0f;		// mm -> m

		if (hh <= 0.0001)
		{
			hh = 0.0f;
		}
		/* ��һ���ر�Ԫ������ϵ��*/
		rman = m_ManningN[rbCell];
		/* ��һ���ر�Ԫ�ںӵ���*/
		if (!FloatEqual(m_streamLink[rbCell], NODATA_VALUE))
		{
			/* ��һ��դ��Ԫ�ϵ���ˮ��� > �ӵ���ȣ����Ϊ��һ���ӵ��ر�Ԫ�ĺӵ����Ѿ�����ˮ*/
			if (m_surSdep[rbCell] > m_chDepth[rbCell])
			{
				/* ��һ���ر�Ԫ�ĵر�ˮ��*/
				stordepth =	m_surSdep[rbCell] - m_chDepth[rbCell];
			}
			else
			{
				stordepth = 0.0f;
			}
		}
		else
		{
			stordepth = m_surSdep[rbCell];
		}
	}
	if (stordepth <= 0.0001)
	{
		stordepth = 0.0f;
	}
	/* ������ȴ��ڵر�ˮ��*/
	// todo ������京�壿
	if (hh >= stordepth)
	{
		/* alfa�Ǹ��������������µ�Ħ��Ƚ����������̬����*/
		alfa = (float)((pow(fabs(sf), 0.5)) / rman);

		/*	Note : The variable "a" represents the sign of the	Friction Slope (Sf)	Computing Overland Flow	*/
		if (sf >= 0) a = 1.0;

		if (sf < 0) a = -1.0;
		/* dqq ʱ�䲽���ڵر������ʵı仯�� = alfa * h��(5/3)�η�*/
		dqq = (float)(a*m_cellWth*alfa*pow((hh - stordepth), 1.667));
		//if (counter >= 600 && Summ_file_fptr.is_open())
		//{
		//	Summ_file_fptr << " dq: " << dqq << " h: " << hh << "  sto�� " << stordepth << " alfa: " << alfa << endl;
		//}
		
		/* ���դ��Ԫ��ʱ�䲽���ڵĵر�������, dqqΪ����ˮ�����ҡ��·���dqqΪ����ˮ���ҡ��·�����ǰ��Ԫ*/
		m_chQ[icell] = m_chQ[icell] - dqq;

		m_chQ[rbCell] = m_chQ[rbCell] + dqq;

	}	/* End of HH >= STORDEPTH */
	//# ifdef IS_DEBUG
	//if (icell == output_icell && Summ_file_fptr.is_open()) {
	//	Summ_file_fptr <<  " exchange: " << -dqq << endl;
	//}
	//else if (rbCell == output_icell && Summ_file_fptr.is_open()) {
	//	Summ_file_fptr << " exchange: " << dqq << endl;
	//}
	//if (isnan(dqq) || isnan(m_chQ[icell]) || isnan(dhdx)) {
	//	if (Summ_file_fptr.is_open()) {
	//		Summ_file_fptr << " icell: " << icell << " dqq: " << dqq << " m_chQ[icell]: " << m_chQ[icell] << " m_chQ[rbCell]: " << m_chQ[rbCell]
	//			<< " hh:" << hh << " stordepth: " << stordepth << " alfa: " << alfa << endl;
	//	}
	//}
	//#endif // IS_DEBUG
	return dqq;
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
	sslope = m_Slope[curCellIndex];					/* side slope				*/
	rmanch = m_ManningN[curCellIndex];		/* manning's n				*/
	//sfactor = m_chSinuosity[curReachIndex];	/* sinuosity factor,use the reach's sinuosity instead of that on every cell of reach temporary	*/
	sfactor = 1;
	stordep = m_surSdep[curCellIndex];			/* Storage depth			*/
	hchan = m_chWtrDepth[curCellIndex];		/* Channel water depth*/
	hh = m_chWtrDepth[curCellIndex] - stordep;
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

	/* Overland water depth at outlet cell is reduced after taking the outflow out of the cell*/
	m_surWtrDepth[m_idOutlet] = (float)(m_surWtrDepth[m_idOutlet] - qoutov * m_dt / (pow(m_cellWth, 2.0)));

	/* SECOND:calculate the flow going out from the channel portion	*/
	if (m_chWtrDepth[m_idOutlet] > m_surSdep[m_idOutlet])
	{
		hout = m_chWtrDepth[m_idOutlet] - m_surSdep[m_idOutlet];

		qoutch = chnDischarge(m_chWtrDepth[m_idOutlet], hout, m_chWidth[m_idOutlet], m_chDepth[m_idOutlet],
			m_surSdep[m_idOutlet], m_ManningN[m_idOutlet], 1, m_Slope[m_idOutlet], 1);

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



