#include "CASC2D_OF.h"
#include "text.h"
using namespace std;

CASC2D_OF::CASC2D_OF() :
    m_nCells(-1),m_nSoilLyrs(nullptr),m_ks(nullptr),m_soilWtrStoPrfl(nullptr),
	m_surWtrDepth(nullptr), m_chWtrDepth(nullptr),m_surSdep(nullptr), m_ManningN(nullptr), m_streamLink(nullptr), m_dem(nullptr) ,
	m_flowOutIndex(nullptr) , m_Slope(nullptr), m_chWidth(nullptr) , m_chSinuosity(nullptr), m_ovQ(nullptr), m_chQ(nullptr), m_outQ(0.0), m_outV(0.0){

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
		*data = m_ovQ;
	}else if (StringMatch(sk, VAR_QRECH)) {
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
	if (nullptr == m_ovQ) Initialize1DArray(m_nCells, m_ovQ, 0.f);
	//if (nullptr == cellWtrDep) Initialize1DArray(30000, cellWtrDep, 0.f);
	// m_surSdep(m)
	if (nullptr == m_surSdep) Initialize1DArray(m_nCells, m_surSdep, 0.f);
	// ��ʼ���ӵ�ˮ��,��ʵ���ｫ�ӵ�����ĳ�ʼˮ���ʱ��Ϊ0
	// �ӵ���ĺӵ���ʼˮ����Ϊ0(m)
	if (nullptr == m_chWtrDepth) Initialize1DArray(m_nCells, m_chWtrDepth, 0.f);
	if (m_InitialInputs) {
		output_icell = 50676;
		printIOvFlowMinT = 239700;
		printOvFlowMaxT = 24630;
		printChFlowMinT = 39000;
		printChFlowMaxT = 39090;
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
	string baseOutputPath = "G:\\program\\\seims\\\data\\log\\";
	// ��������
	std::ostringstream ovflowOss;
	ovflowOss << baseOutputPath << "ov_flow_" << counter << ".txt";
	string ovFlowFile = ovflowOss.str();

	// �ӵ�����
	std::ostringstream chflowOss;
	chflowOss << baseOutputPath << "ch_flow_" << counter << ".txt";
	string chFlowFile = chflowOss.str();

	// ����ˮ���쳣ֵ
	std::ostringstream wtrDepOss;
	wtrDepOss << baseOutputPath << "ov_wtrdep" <<  ".txt";
	string wtrDepFile = wtrDepOss.str();
	if (counter == 0)
	{
		deleteExistFile(wtrDepFile);
		if (!wtrDepFptr.is_open())
		{
			wtrDepFptr.open(wtrDepFile.c_str(), std::ios::out | std::ios::app);
		}
	}
	wtrDepFptr << counter << endl;

	// ���ÿ��դ��Ԫ������λ��
	std::ostringstream positionOss;
	positionOss << baseOutputPath << "position.txt";
	string positionFile = positionOss.str();
	// ����ӵ�����
	if ((counter >= printChFlowMinT && counter <= printChFlowMaxT) && !chFlowFptr.is_open())
	{
		deleteExistFile(chFlowFile);
		chFlowFptr.open(chFlowFile.c_str(), std::ios::out | std::ios::app);
	}
	//���λ������
	if (counter == 0) {
		//deleteExistFile(positionFile);
		//position_Fptr.open(positionFile.c_str(), std::ios::out | std::ios::app);
		//printPosition();
		// ����λ��������dqq����
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

	# ifdef IS_DEBUG
	// �����������
	if ((counter >= printIOvFlowMinT && counter <= printOvFlowMaxT) && !ovFlowFptr.is_open())
	{
		deleteExistFile(ovFlowFile);
		ovFlowFptr.open(ovFlowFile.c_str(), std::ios::out | std::ios::app);
		printOvFlow();
		if (ovFlowFptr.is_open()) {
			ovFlowFptr.close();
		}
	}
	if (chFlowFptr.is_open()) {
		chFlowFptr.close();
	}
	if (counter >= 25000 && wtrDepFptr.is_open()) {
		chFlowFptr.close();
	}
	
	# endif
	double sub_t2 = TimeCounting();
	cout << "casc2d_sed timestamp  end, cost time: " << sub_t2 - sub_t1 << endl;

	return 0;
}

void CASC2D_OF::deleteExistFile(string file) {
	if (_access(file.c_str(), 0) == 0) {
		if (remove(file.c_str()) == 0) {
			cout << "succeed to delete casc2d output file " << file.c_str() << endl;
		}
		else {
			cout << "failed to delete casc2d output file.  " << file.c_str() << endl;
		}
	}
}

// ������ж�Ӧ��iCell��Ϣ
void CASC2D_OF::printPosition() {
	int last_row = -1;
	int last_col = -1;
	int start_col = 0;
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		if (iCell == 0)
		{
			for (int i = 1; i <= m_ncols; i++)
			{
				position_Fptr << std::left << setw(7) << setfill(' ') << i;
			}
		}
		// ����
		if (last_row != curRow)
		{
			position_Fptr << endl;
			// �������
			for (int i = 1; i < curCol; i++)
			{
				position_Fptr << setfill(' ') << setw(7) << ' ';
			}
			position_Fptr << std::left << setw(7) << setfill(' ') << iCell;
		}
		else
		{
			// �����ǰ��Ԫ���ϸ���Ԫ�Ҳ���ڵģ���ֱ�������ǰ��Ԫ
			if (iCell == 0 || curCol == last_col + 1) {
				position_Fptr << std::left << setw(7) << setfill(' ') << iCell;
			}
			else
			{
				// �����ǰ��Ԫ���ϸ���Ԫ֮���м�����������Щ������������ǰ��Ԫ
				for (int i = last_col + 1; i < curCol; i++)
				{
					position_Fptr << setfill(' ') << setw(7) << ' ';
				}
				position_Fptr << std::left << setw(7) << setfill(' ') << iCell;
			}
		}
		last_row = curRow;
		last_col = curCol;
	}
}

void CASC2D_OF::printOvFlow() {
	int last_row = -1;
	int last_col = -1;
	int start_col = 0;
	int cols = 0;
	//// ��¼ÿ�еĵ�1�к����һ�е��к�
	int lastrow_firstcol = 0;
	int lastrow_lastcol = 0;
	int lastrow_firstcell = 0;
	int lastrow_lastcell = 0;
	// ��¼ÿ�е���Щ�в�Ϊ��
	//int *rowMask = new int[m_ncols];
	for (int iCell = 0; iCell < m_nCells; iCell++) {
		int curRow = m_RasterPostion[iCell][0];
		int curCol = m_RasterPostion[iCell][1];
		// ��һ�д�ӡ�кţ������Ŵ�ӡ���к�����
		if (iCell == 0)
		{
			for (int i = 1; i <= m_ncols; i++)
			{
				ovFlowFptr << std::left << setfill(' ') << setw(35) << i;
			}
			lastrow_firstcol = curCol;
			// ��ӡ���к�����
			printLineBreak(lastrow_firstcol);
			printCellFlow(iCell);
		}
		// �ӵڶ��п�ʼ��ÿ�λ��ж���ӡ���з���������y������������һ�еĵ�һ��Ԫ��
		else if (last_row != curRow)
		{
			// ��ӡ���к�����
			printLineBreak(lastrow_firstcol);
			// ��¼��һ�����1�еĵ�Ԫ��
			lastrow_lastcell = iCell - 1;
			// ��¼��һ�����1�е��к�
			lastrow_lastcol = m_RasterPostion[iCell - 1][1];
			// ���y����������dqq>0��������
			int last_col = 0;
			int cur_col = 0;
			// ��ӡ��һ�е�1�е�y��������
			printArrow(iCell);
			// ��ӡ��һ�еڶ��п�ʼ��y��������
			for (int i = lastrow_firstcell + 1; i <= lastrow_lastcell; i++)
			{
				int cur_col = m_RasterPostion[i][1];
				int last_col = m_RasterPostion[i - 1][1];
				printCellArrow(i, last_col, cur_col);
			}
			cols = 0;
			// ��ӡ���к�����
			printLineBreak(curCol);
			// ��ӡ��һ�е�һ����Ԫ������
			printCellFlow(iCell);
			// ��¼��һ�еĵ�1�е��к�
			lastrow_firstcol = curCol;
			// ��¼��һ�еĵ�1�еĵ�Ԫ��
			lastrow_firstcell = iCell;
		}
		// �ǻ���
		else
		{
			// �����ǰ��Ԫ���ϸ���Ԫ�Ҳ���ڵģ���ֱ�������ǰ��Ԫ
			if (curCol == last_col + 1) {
				printCellFlow(iCell);
			}
			else
			{
				// �����ǰ��Ԫ���ϸ���Ԫ֮���м�����������Щ������������ǰ��Ԫ
				for (int i = last_col + 1; i < curCol; i++)
				{
					ovFlowFptr << setfill(' ') << setw(35) << ' ';
				}
				printCellFlow(iCell);
			}
			//rowMask[cur_col] = 1;
			//lineEnd = cur_col;
		}

		cols++;
		last_row = curRow;
		last_col = curCol;
	}
}

void CASC2D_OF::printChFlow() {
	// todo ��ӡÿ��ʱ�䲽���ĺӵ�����
}

void CASC2D_OF::printLineBreak(int lastrow_firstcol) {
	ovFlowFptr << endl;
	// �������
	for (int i = 1; i < lastrow_firstcol; i++)
	{
		ovFlowFptr << setfill(' ') << setw(35) << ' ';
	}
}

void CASC2D_OF::printCellFlow(int iCell) {
	// ��ӡλ��
	ovFlowFptr << std::left << setw(7) << setfill(' ') << iCell;
	// ��ӡˮ��
	ovFlowFptr << std::left << setw(12) << setfill(' ') << m_surWtrDepth[iCell];
	// ��ӡx��������,dqq>0������
	if (m_Dqq[iCell][0] > 0)
	{
		ovFlowFptr << "��" << std::left << setw(15) << setfill(' ') << m_Dqq[iCell][0];
	}
	else if (m_Dqq[iCell][0] == 0) {
		ovFlowFptr << "--" << std::left << setw(14) << setfill(' ') << m_Dqq[iCell][0];
	}
	else
	{
		ovFlowFptr << "��" << std::left << setw(15) << setfill(' ') << -m_Dqq[iCell][0];
	}
}

void CASC2D_OF::printCellArrow(int iCell, int last_col, int cur_col) {
	// ��ʱ�������м��м�������
	//printArrow(iCell);
	// �����м��м�������
	if (cur_col == last_col + 1)
	{
		printArrow(iCell);
	}
	else
	{
		for (int i = last_col + 1; i < cur_col; i++)
		{
			ovFlowFptr << setfill(' ') << setw(35) << ' ';
		}
		printArrow(iCell);
	}

}

void CASC2D_OF::printArrow(int iCell) {
	if (m_Dqq[iCell][1] > 0)
	{
		ovFlowFptr << "��" << std::left << setw(15) << setfill(' ') << m_Dqq[iCell][1];
	}
	else if (m_Dqq[iCell][1] == 0)
	{
		ovFlowFptr << "|" << std::left << setw(15) << setfill(' ') << m_Dqq[iCell][1];
	}
	else {
		ovFlowFptr << "��" << std::left << setw(15) << setfill(' ') << -m_Dqq[iCell][1];
	}
	ovFlowFptr << setfill(' ') << setw(19) << ' ';

}

// ����λ��������dqq���飬�������� ��Ӧ 0123����dqq��Ӧ4����dqq��Ӧ5
void CASC2D_OF::buildPositionIndex() {
	// λ��
	m_RasterNeighbor = new int* [m_nCells];
	m_Dqq = new float*[m_nCells];
	for (int i = 0; i < m_nCells; i++) {
		m_RasterNeighbor[i] = new int[4];
		m_Dqq[i] = new float[2];
		for (int j = 0; j < 6; j++)
		{
			m_RasterNeighbor[i][j] = -1;
		}
		for (int j = 0; j < 2; j++)
		{
			m_Dqq[i][j] = 0.0;
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
		hov = m_ovQ[i] * m_dt / (m_cellWth*m_cellWth);
		/*�߼������h�ĸ���ɾȥ���ꡢ������������Ӱ�죬��Ϊ����ģ���Ѿ����������������Ǿ����ٶ������ˮ��仯*/
		hov = hov + m_surWtrDepth[i] / 1000.f; //  mm -> m
		if (hov < 0.0)
		{
			hov = 0.0f;
		}
		/* ����ȫ�ֵر�ˮ��*/
		m_surWtrDepth[i] = hov * 1000.f;			// m -> mm
		/* ����ǰʱ�䲽���ڵĵر����ٱ仯��Ϊ0*/
		m_ovQ[i] = 0.0;
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

		//#endif // IS_DEBUG
		map<int,vector<int>>::iterator it;
		it = m_rbcellsMap.find(iCell);
		if (it != m_rbcellsMap.end())
		{
			float dqq;
			int rightCell;
			// ����ҷ�դ��Ԫ��Ϊ�գ������ҷ�դ��Ԫ
			if ((it->second)[0] != -1)
			{
				rightCell = (it->second)[0];
				dqq = ovrl(iCell, rightCell);
				# ifdef IS_DEBUG
				m_Dqq[iCell][0] = dqq;
				#endif // IS_DEBUG
				
			}

			int belowCell;
			// ����·�դ��Ԫ��Ϊ�գ������·�դ��Ԫ
			if ((it->second)[1] != -1)
			{
				belowCell = (it->second)[1];
				dqq = ovrl(iCell, belowCell);
				# ifdef IS_DEBUG
				m_Dqq[iCell][1] = dqq;
				#endif // IS_DEBUG
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
	//	if (ovFlowFptr.is_open()) {
	//		ovFlowFptr << " icell: " << icell << " m_surWtrDepth[icell]: " << m_surWtrDepth[icell] << " m_surWtrDepth[rbCell]: " << m_surWtrDepth[rbCell] << endl;
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
		float newH = hh - stordepth;
		if (newH < 0.0)
		{
			newH = 0.0;
		}
		// todo: �����dqq����Ҫ��Ҫ����dt
		//dqq = (float)(a*m_cellWth*alfa*pow((newH), 1.667)) / m_dt;
		dqq = (float)(a*m_cellWth*alfa*pow((newH), 1.667)) ;
		# ifdef IS_DEBUG
		if (isnan(dqq) || isinf(dqq) || isnan(dhdx) || isinf(dhdx) || isnan(sf) || isinf(sf))
		{
			wtrDepFptr << "m_surWtrDepth[" << icell << "]: " << m_surWtrDepth[icell] << " m_surWtrDepth[" << rbCell << "]: " << m_surWtrDepth[rbCell]
				<< " sf: " << sf << " so: " << so << " dhdx: " << dhdx << " rman: " << "dqq: " << dqq
				<< " alfa: " << alfa << " hh - stordepth: " << hh - stordepth << endl;
		}
		#endif // IS_DEBUG
		/* ���դ��Ԫ��ʱ�䲽���ڵĵر�������, dqqΪ����ˮ�����ҡ��·���dqqΪ����ˮ���ҡ��·�����ǰ��Ԫ*/

		m_ovQ[icell] = m_ovQ[icell] - 0.5 * dqq;

		m_ovQ[rbCell] = m_ovQ[rbCell] + 0.5 * dqq;

	}	/* End of HH >= STORDEPTH */
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
						sdep_ov = m_surSdep[idCell]  - dch * 1000.0f;
					else
						sdep_ov = 0.0;
					/* ��ǰʱ�䲽�����Եر�Ԫ��ˮ�����*/
					vol_ov_in = 0;
					float wtrDepthTmp = m_surWtrDepth[idCell];
					if (m_surWtrDepth[idCell]  > sdep_ov)
					{
						// �ر�ˮ������ӵ��󣬽��ر�ˮ������Ϊ�ر��ʼ�ȶ�ˮ����ȣ�0��
						vol_ov_in = (m_surWtrDepth[idCell]  - sdep_ov) / 1000.0f *m_cellWth*m_cellWth;
						m_surWtrDepth[idCell] = sdep_ov;
					}
					# ifdef IS_DEBUG
					if (counter >= printChFlowMinT && counter <= printChFlowMaxT) {
						chFlowFptr << "RCH_ID: " << std::left << setw(4) << setfill(' ') << reachId << " "
							<< "DOWN_ID: " << std::left << setw(4) << setfill(' ') << m_downStreamReachId[reachId] << " "
							<< "CELL_ID: " << std::left << setw(8) << setfill(' ') << idCell << " "
							<< "S_DEP: " << std::left << setw(8) << setfill(' ') << wtrDepthTmp << " "
							<< "WCH: " << std::left << setw(6) << setfill(' ') << fixed << setprecision(3) << wch << " "
							<< "DCH: " << std::left << setw(6) << setfill(' ') << fixed << setprecision(3) << dch << " "
							<< "CH_Q: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << m_chQ[idCell] << " "
							<< "CH_IN: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << inflowVol << " "
							<< "OV_IN: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << vol_ov_in << " ";
					}
					# endif
					m_chWtrDepth[idCell] = newChnDepth(wch, dch, sfactor, idCell, (inflowVol + vol_ov_in));   // m


					
					/* Negative Depth in the Channel --> EXIT program					*/

					if (m_chWtrDepth[idCell] < 0.0)
					{
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
	# ifdef IS_DEBUG
	if (counter >= printChFlowMinT && counter <= printChFlowMaxT) {

		chFlowFptr
			<< "VOL_ADD: " << std::left << setw(13) << setfill(' ') << fixed << setprecision(3) << addedVolume
			<< "AREA_INIT: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << area_init
			<< "VOL_INIT: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << vol_init
			<< "VO_FIN: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << vol_final
			<< "OLD_DEP: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << m_chWtrDepth[idCell]
			<< "NEW_DEP: " << std::left << setw(10) << setfill(' ') << fixed << setprecision(3) << newdepth << endl;
			;
	}
	# endif
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
				//if (counter == 26010 && reachId == 35)
				//{
				//	cout << endl;
				//}
				for (int iCell = 0; iCell < n; ++iCell) {
					int cellIndex = curReachCells[iCell];
					chnchn
					(reachIndex, reachId, nReaches, iCell, curReachCells);
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
		sf = (float)(so + dhdx + 1e-30);

		/* Nota de Jorge: Sf se deberia quedar con el mismo signo */
		/* sf Ӧ�ñ�����ͬ�ķ��ţ���sf����ֵ�����С������sfΪ��ֵ*/
		if (fabs(sf) < 1e-20) sf = (float)(1e-20);

		if (sf < 0.0)
		{
			// �ӵ���ֻ������ǰ��
			/*a = (float)(-1.0*a);*/
			a = (float)(1.0*a);

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
			(float)(m_cellWth*alfa*pow((m_surWtrDepth[m_idOutlet] / 1000.0 - m_surSdep[m_idOutlet]), 1.667));
	}

	/* Overland water depth at outlet cell is reduced after taking the outflow out of the cell*/
	m_surWtrDepth[m_idOutlet] = (float)((m_surWtrDepth[m_idOutlet] / 1000.0 - qoutov * m_dt / (pow(m_cellWth, 2.0)))) * 1000.0;

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
		// �ӵ�������� + �ӵ��ϵ����ϵĲ������
		area = wch * (dch - stordep) + m_cellWth * (hchan - dch);
		// �ӿ� + 2* �ӵ���� + 2*��դ��� - �ӿ�+ 2*��ˮ�� - �ӵ��
		wp = (float)(wch + 2 * (dch - stordep) + 2 * (m_cellWth - wch) + 2 * (hchan - dch));
	}

	//dQ = (float)(a*(sqrt(fabs(sf)) / rmanch)*
	//	(pow(area, 1.6667)) / (pow(wp, 0.6667))) / m_dt;
	dQ = (float)(a*(sqrt(fabs(sf)) / rmanch)*
		(pow(area, 1.6667)) / (pow(wp, 0.6667))) ;

	/* Limit the outflow by availability														*/
	/* ������������*/
	vol_ch_avail = area * m_cellWth * sfactor;

	if (dQ*m_dt > vol_ch_avail) dQ = vol_ch_avail / m_dt;

	return(dQ);

}



