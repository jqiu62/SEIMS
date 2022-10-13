#include "CASC2D_OF.h"
#include "text.h"

CASC2D_OF::CASC2D_OF() :
    m_nCells(-1),m_nSoilLyrs(nullptr),m_ks(nullptr),m_soilWtrStoPrfl(nullptr),
	m_surfRf(nullptr), m_surSdep(nullptr), m_ManningN(nullptr), m_streamLink(nullptr), m_dem(nullptr) ,
	m_flowOutIndex(nullptr) , m_Slope(nullptr), m_chWidth(nullptr) {

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
	}
}

void CASC2D_OF::SetValueByIndex(const char* key, int index, float value) {
}

void CASC2D_OF::Set1DData(const char* key, int n, float* data) {
	/* todo ���ⲿ���յĽ���ǿ��һάת��ά����ֵ��m_rint** */
	if (!CheckInputSize("CASC2D_OF", key, n, m_nCells)) return;
	string sk(key);
	 if (StringMatch(sk, VAR_SURU)) {
		m_surfRf = data;  
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
		 m_reachDepth = data;
	}else
	{
		throw ModelException("CASC2D_OF", "Set1DData", "parameter " + string(key) + " is not exist");
	}
}

void CASC2D_OF::Set2DData(const char* key, int n, int col, float** data) {
	
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

void CASC2D_OF::SetScenario(Scenario* sce) {
}

bool CASC2D_OF::CheckInputData() {
    return true;
}


void CASC2D_OF::InitialOutputs() {
	if (nullptr == m_chQ) Initialize1DArray(m_nCells, m_chQ, 0.f);
	if (nullptr == m_surSdep) Initialize1DArray(m_nCells, m_surSdep, 0.f);
}


int CASC2D_OF::Execute() {
	InitialOutputs();
	/* Overland and channel depth updating												*/
	OvrlDepth();
	ChannDepth();
	/* Overland and channel flow routing												*/
	OvrlRout();
	ChannRout();

	/* Flow routing at the outlet																*/
	RoutOutlet();
    return 0;
}

TimeStepType CASC2D_OF::GetTimeStepType() {
    return TIMESTEP_HILLSLOPE;
}



void CASC2D_OF::GetValue(const char* key, float* value) {
}

void CASC2D_OF::Get1DData(const char* key, int* n, float** data) {
	string sk(key);
	if (StringMatch(sk, VAR_QOVERLAND)) {
		*data = m_chQ;
	}
}

void CASC2D_OF::Get2DData(const char* key, int* n, int* col, float*** data) {	//const char* key��keyָ������ݲ��ܸı�
}

void CASC2D_OF::SetRasterPositionDataPointer(const char* key, int** positions) {
	m_RasterPostion = positions;
}

//void CASC2D_OF::SetReachDepthData( FloatRaster* positions) {
//	m_reachDepth = positions;
//}


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
		hov = hov + m_surfRf[i];
		/* ����ȫ�ֵر�ˮ��*/
		m_surfRf[i] = hov;
		/* ����ǰʱ�䲽���ڵĵر����ٱ仯��Ϊ0*/
		m_chQ[i] = 0.0;
	}
	
}

/*************************���ºӵ��������*******************************/
void CASC2D_OF::ChannDepth()
{
	int ic, j, l, k, jj;
	float wch, dch, sfactor, sdep_ov, inflowVol, vol_ov_in;
	for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
		int nReaches = it->second.size();
		for (int i = 0; i < nReaches; ++i) {
			int reachIndex = it->second[i]; // index in the array, from 0
			vector<int> &vecCells = m_reachs[reachIndex];
			int n = vecCells.size();
			for (int iCell = 0; iCell < n; ++iCell) {
				int idCell = vecCells[iCell];
				/* �ӵ���ȡ���ȡ�������*/
				wch = m_chWidth[idCell];
				dch = m_chDepth[idCell];
				sfactor = m_chSinuosity[idCell];			
				/* Find new channel depth after adding inflow volume				*/
				/* ��ǰʱ�䲽�����������*/
				inflowVol = m_chQ[idCell] * m_dt;
				/* ... and the volume coming from the overland	(vol_ov_in) */
				/* �����ӵ���ȵ�ˮ�����*/
				if (m_chQ[idCell] > dch)
					sdep_ov = m_chQ[idCell] - dch;
				else
					sdep_ov = 0.0;

				vol_ov_in = 0;

				if (m_surfRf[idCell] > sdep_ov)
				{
					vol_ov_in = (m_surfRf[idCell] - sdep_ov)*m_cellWth*m_cellWth;
					m_surfRf[idCell] = sdep_ov;
				}

				m_chWtrDepth[idCell] = newChnDepth(wch, dch, sfactor, idCell,
					(inflowVol + vol_ov_in));


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

/*************************�������*******************************/
void CASC2D_OF::OvrlRout()
{

	int j, k, jj, kk, l;
	/* ����դ��Ԫ*/
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
		for (int iPosition = 0; iPosition < m_nCells; iPosition++)
		{
			if (m_RasterPostion[iPosition][0] == rightRow && m_RasterPostion[iPosition][1] == rightCol) {
				isRightExists = true;
				rightCell = iPosition;
			}
			if (m_RasterPostion[iPosition][0] == belowRow && m_RasterPostion[iPosition][1] == belowCol) {
				isBelowExists = true;
				belowCell = iPosition;
			}
		}
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
	
	}
}

void CASC2D_OF::ovrl(int icell, int rbCell)
{
	int jfrom, kfrom, jto, kto;

	float a = 1.0;

	float vel = 0.0;

	float so, sf, dhdx, hh, rman, alfa, dqq, stordepth;

	so = (m_dem[icell] - m_dem[rbCell]) / m_cellWth;			/* �Ӵ��¶�*/

	dhdx = (m_surfRf[rbCell] - m_surfRf[icell]) / m_cellWth;		/* dh/dx, �ر�ˮ��/դ��Ԫ��ȣ����ر�ˮ���¶�*/

	sf = so - dhdx + (float)(1e-30);	/* Ħ���¶�*/

	hh = m_surfRf[icell];                  /* �ر���ˮ��*/

	rman = m_ManningN[icell];		 /* ����ϵ��*/

	/* �ںӵ���*/
	//if (chancheck == 1 && link[j][k] > 0)
	if (!FloatEqual(m_streamLink[icell], NODATA_VALUE))
	{
		/* ���դ��Ԫ�ϵ���ˮ��� > �ӵ���ȣ���ӵ��Ϸ��ĵر�ˮ��=դ��Ԫ�ϵ�����ˮ���-�ӵ���ȣ�����ر�ˮ��=0.0*/
		if (m_surSdep[icell] > m_chDepth[icell])
		{
			/* �ر�ˮ�� = �ӵ���դ��Ԫ�ϵ�����ˮ��� - �ӵ���ȣ����߳��ӵ����ֵ���ȣ�*/
			stordepth = m_surSdep[icell] - m_chDepth[icell];
		}
		else
		{
			stordepth = 0.0;
		}
	}
	/* �ںӵ��⣬�ر�ˮ�� = դ��Ԫ�ϵ���ˮ���*/
	else
	{
		stordepth = m_surSdep[icell];
	}

	/* Ħ���Ƚ� < 0��˵������դ��Ԫ֮���ˮ���ٶȷ����ı䣬���Ҫ�����¸�դ��Ԫ�ĵر�ˮ��*/
	if (sf < 0)
	{
		/* ��һ���ر�Ԫ�ĵر���ˮ��*/
		hh = m_surfRf[rbCell];
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
					m_surSdep[rbCell] - m_chDepth[rbCell];
			}
			else
			{
				stordepth = 0.0;
			}
		}
		else
		{
			stordepth = m_surSdep[rbCell];
		}
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

		/* ���դ��Ԫ��ʱ�䲽���ڵĵر�������*/
		m_chQ[icell] = m_chQ[icell] - dqq;

		m_chQ[rbCell] = m_chQ[rbCell] + dqq;

	}	/* End of HH >= STORDEPTH */

}   /* End of OVRL */

/*************************�ӵ�����*******************************/
void CASC2D_OF::ChannRout()
{
	for (auto it = m_reachLayers.begin(); it != m_reachLayers.end(); it++) {
		int nReaches = it->second.size();
		for (int i = 0; i < nReaches; ++i) {
			int reachIndex = it->second[i]; // index in the array, from 0
			vector<int> &curReachCells = m_reachs[reachIndex];
			int n = curReachCells.size();
			for (int iCell = 0; iCell < n; ++iCell) {
				int cellIndex = curReachCells[iCell];
				chnchn(reachIndex, nReaches,iCell, curReachCells);
			}
		}
	}
}
/*
curReachIndex: �ӵ����
iCell: �ӵ��еĵ�iCell���ڵ�
curCellIndex: �ӵ��еĵ�iCell���ڵ���դ�������е��±�
*/
void CASC2D_OF::chnchn(int curReachIndex,int nReaches,int iCell, vector<int> curReachCells)
{

	float a = 1.0;
	float vel = 0.0;
	float wch, dch, sslope, rmanch, sfactor, so, sf, hh, dhdx,dq, stordep, hchan, dq_ov;

	int  j, k, jj, kk, jjj, iic, ijun, ill;
	int curCellIndex = curReachCells[iCell];
	/* row and column of following node in link IC									*/
	/* index of following node in link curReachIndex*/
	// ��ǰ�ӵ���ǰ�ڵ����һ���ڵ�ĵ��
	int nextCellIndex = curReachCells[iCell + 1];

	/* Note : JJJ is a check to see when the end of									*/
	/*        a channel link has been reached.											*/

	//int nextNextCell = m_reachs[curReachIndex][iCell + 2];
	//int nextReachId = m_downStreamReachId.
	/* Channel characteristics :																		*/

	wch = m_chWidth[curCellIndex];				/* width						*/
	dch = m_reachDepth[curCellIndex];
	//dch = m_reachDepth->GetRasterDataPointer()[curCellIndex];				/* depth						*/
	sslope = m_Slope[curCellIndex];					/* side slope					*/
	rmanch = m_ManningN[curCellIndex];		/* manning's n				*/
	sfactor = m_chSinuosity[curReachIndex];	/* sinuosity factor,use the reach's sinuosity instead of that on every cell of reach temporary	*/
	stordep =	m_surSdep[curCellIndex];			/* Storage depth			*/
	hchan = m_chWtrDepth[curCellIndex];		/* Channel water depth*/
	hh = m_chWtrDepth[curCellIndex] - stordep;

	/* Channel slope																								*/

	so = (m_dem[curCellIndex] - m_dem[nextCellIndex]) / (m_cellWth*sfactor);

	dq_ov = 0.0;
	/* judge whether the next cell is the last cell of current reach*/
	if (true)
	{
		/* �������кӵ������ĳ�ӵ��ĵ�һ���ڵ�����к� == ��ǰ�ӵ���ǰ�ڵ����һ���ڵ�����кţ�
		˵����ǰ�ڵ��ǵ�ǰ�ӵ������һ���ڵ�*/
		for (size_t reachIndex = 0; reachIndex < nReaches; reachIndex++)
		{
			vector<int> reachCells = m_reachs[reachIndex];
			if (nextCellIndex == reachCells[0])
			{
				//so = (m_dem[curCellIndex] - dch - m_dem[nextCellIndex] + m_reachDepth->GetRasterDataPointer()[reachCells[0]]) / (m_cellWth*sfactor);
				so = (m_dem[curCellIndex] - dch - m_dem[nextCellIndex] + m_reachDepth[reachCells[0]]) / (m_cellWth*sfactor);
				
				ijun = reachIndex;
			}
		}

	}
	/* hch[j][k]դ��Ԫ�ϵĺӵ�ˮ�dhdx��ˮ���¶�*/
	//dhdx = (m_reachDepth->GetRasterDataPointer()[nextCellIndex] - m_reachDepth->GetRasterDataPointer()[curCellIndex] / (m_cellWth*sfactor));
	dhdx = (m_reachDepth[nextCellIndex] - m_reachDepth[curCellIndex] / (m_cellWth*sfactor));

	/* Ħ���¶�*/
	sf = (float)(so - dhdx + 1e-30);

	/* Nota de Jorge: Sf se deberia quedar con el mismo signo */
	/* sf Ӧ�ñ�����ͬ�ķ��ţ���sf����ֵ�����С������sfΪ��ֵ*/
	if (fabs(sf) < 1e-20) sf = (float)(1e-20);

	if (sf < 0.0)
	{
		a = (float)(-1.0*a);

		/* �����ǰ�ڵ��ǵ�ǰ�ӵ������һ���ڵ㣬���ȡ���κӵ��ĵ�һ���ڵ������*/
		if (jjj < 0)
		{
			/* Take channel chars. of the 1st node of downstream link */
			int nextReachIndex = m_downStreamReachId[curReachIndex];
			vector<int> nextReachCells = m_reachs[nextReachIndex];
			int nextReachFistCellIndex = nextReachCells[0];
			wch = m_chWidth[nextReachFistCellIndex];
			dch = m_reachDepth[nextReachFistCellIndex];
			sslope = m_Slope[nextReachFistCellIndex];
			rmanch = m_ManningN[nextReachFistCellIndex];
			sfactor = m_chSinuosity[nextReachIndex];
		}
		/*�����ǰ�ڵ㲻�ǵ�ǰ�ӵ������һ���ڵ㣬���ȡ��ǰ�ӵ�����һ���ڵ�*/
		else
		{
			/*Take channel chars. of the next node within current link */

			wch = m_chWidth[nextCellIndex];
			dch = m_reachDepth[nextCellIndex];
			sslope = m_Slope[nextCellIndex];
			rmanch = m_ManningN[nextCellIndex];
			sfactor = m_chSinuosity[curReachIndex];
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
	alfa = (float)(sqrt(sovout) / pman[iman[jout][kout]]);

	/* Discharge from overland flow.  NOTE: because the water from  */
	/* this part of the outlet overland cell was already "poured"		*/
	/* into the channel when updating the channel depth (channDepth)*/
	/* qoutov = 0 when the channel routing is selected							*/

	if (h[jout][kout] > sdep[jout][kout])
	{
		qoutov =
			(float)(w*alfa*pow((h[jout][kout] - sdep[jout][kout]), 1.667));
	}

	/* Overland water depth at outlet cell is reduced after taking	*/
	/* the outflow out of the cell																	*/

	h[jout][kout] = (float)(h[jout][kout] - qoutov * m_dt / (pow(w, 2.0)));

	/* SECOND:calculate the flow going out from the channel portion	*/
	//if (chancheck == 1 && hch[jout][kout] > sdep[jout][kout])
	if (hch[jout][kout] > sdep[jout][kout])
	{
		hout = hch[jout][kout] - sdep[jout][kout];

		qoutch = chnDischarge(hch[jout][kout], hout, wchout, dchout,
			sdep[jout][kout], rmanout, 1, sout, sfactorout);

		dqch[jout][kout] = dqch[jout][kout] - qoutch;
	}

	/* The total outflow at the basin's outlet is given by adding		*/
	/* the outflow from the overland & channel portion of the cell	*/

	qout = qoutov + qoutch;

	/* Keeping Track of the Total Outflow Volume										*/

	//vout = vout + qout * dt;

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
		area = wch * (dch - stordep) + w * (hchan - dch);
		wp = (float)(wch + 2 * (dch - stordep) + 2 * (w - wch) + 2 * (hchan - dch));
	}

	dQ = (float)(a*(sqrt(fabs(sf)) / rmanch)*
		(pow(area, 1.6667)) / (pow(wp, 0.6667)));

	/* Limit the outflow by availability														*/
	/* ������������*/
	vol_ch_avail = area * w * sfactor;

	if (dQ*m_dt > vol_ch_avail) dQ = vol_ch_avail / m_dt;

	return(dQ);

}



