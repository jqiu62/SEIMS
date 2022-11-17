#include "DepressionFS.h"
#include "text.h"

DepressionFS::DepressionFS(void) : m_nCells(-1),
									//xdw�޸�, DEP_FSģ��û���õ�PET��ɢ���ݣ���ʱע�͵�
                                   //m_depCo(NODATA_VALUE), m_depCap(NULL), m_pet(NULL), m_ei(NULL),
									m_depCo(NODATA_VALUE), m_depCap(NULL),  m_ei(NULL),
                                   m_sd(NULL), m_sr(NULL), m_checkInput(true) {
}

DepressionFS::~DepressionFS(void) {
    Release1DArray(m_sd);
    Release1DArray(m_sr);
    Release1DArray(m_storageCapSurplus);
}

bool DepressionFS::CheckInputData(void) {
    if (m_date == -1) {
        throw ModelException(MID_DEP_FS, "CheckInputData", "You have not set the time.");
    }
    if (m_nCells <= 0) {
        throw ModelException(MID_DEP_FS, "CheckInputData", "The cell number of the input can not be less than zero.");
    }
    if (m_depCo == NODATA_VALUE) {
        throw ModelException(MID_DEP_FS, "CheckInputData",
                             "The parameter: initial depression storage coefficient has not been set.");
    }
    if (m_depCap == NULL) {
        throw ModelException(MID_DEP_FS, "CheckInputData",
                             "The parameter: depression storage capacity has not been set.");
    }
#ifndef STORM_MODE
	//xdw�޸�, DEP_FSģ��û���õ�PET��ɢ���ݣ���ʱע�͵�
    //if (m_pet == NULL) {
    //    throw ModelException(MID_DEP_FS, "CheckInputData", "The parameter: PET has not been set.");
    //}
    //if (m_ei == NULL) {
    //    throw ModelException(MID_DEP_FS, "CheckInputData",
    //                         "The parameter: evaporation from the interception storage has not been set.");
    //}
#endif /* not STORM_MODE */
    return true;
}

void DepressionFS:: InitialOutputs() {
    if (m_nCells <= 0) {
        throw ModelException(MID_DEP_FS, "initialOutputs", "The cell number of the input can not be less than zero.");
    }
    if (m_sd == NULL) {
        m_sd = new float[m_nCells];//�ݵ���ˮ��� mm
        m_sr = new float[m_nCells];//�ر������ mm
        m_storageCapSurplus = new float[m_nCells];// ʣ��洢���(�ݵ�ˮ��������ݵ��ϱ���ľ���)
//#pragma omp parallel for
        for (int i = 0; i < m_nCells; ++i) {
            m_sd[i] = m_depCo * m_depCap[i];// �ݵ���ˮ��� = �ݵس�ʼ��ˮϵ�� * �ݵ����
            m_sr[i] = 0.0f;
        }
		cout << endl;
    }
}

int DepressionFS::Execute() {
    InitialOutputs();
    if (m_checkInput) {
        CheckInputData();
        m_checkInput = false;
    }

//#pragma omp parallel for
    for (int i = 0; i < m_nCells; ++i) {

        // sr is temporarily used to stored the water depth including the depression storage
		// m_sr�ǵر�ˮ��ݵ�����֮ǰ��m_sr�������ݵ�ˮ�
        float hWater = m_sr[i];
		// ��� �ر�ˮ�� <= �ݵ����
		// �ݵ�����֮ǰ
        if (hWater <= m_depCap[i]) {
			// �ݵ�ˮ�� = �ر�ˮ��
            m_sd[i] = hWater;
			// �ر�ˮ�� = 0
            m_sr[i] = 0.f;						
        } else {
			// �ݵ�����֮��
			// �ݵ�ˮ�� = �ݵ����
            m_sd[i] = m_depCap[i];
			// �ر�ˮ�� = �ر�ˮ�� - �ݵ���ȣ���ʱ�ĵر�ˮ��ָ�ݵ��ϱ������ϵ�ˮ��
            m_sr[i] = hWater - m_depCap[i];	
        }
		// ʣ��洢���� = �ݵ���� - �ݵ�ˮ��
        m_storageCapSurplus[i] = m_depCap[i] - m_sd[i];
		//if (i % 100 == 0)
		//{
		//	cout << "hWater: " << hWater << " m_sr: " << m_sr[i] << " m_depCap: " << m_depCap[i] << " m_storageCapSurplus: " << m_storageCapSurplus[i] << endl;
		//}
		//if (m_sr[i] < 0.000001)
		//{
		//	m_sr[i] = 0.f;
		//}
		//if (m_sd[i] < 0.000001)
		//{
		//	m_sd[i] = 0.f;
		//}
		//if (m_storageCapSurplus[i] < 0.000001)
		//{
		//	m_storageCapSurplus[i] = 0.f;
		//}
		//if (m_depCap[i] < 0.000001)
		//{
		//	m_depCap[i] = 0.f;
		//}
    }
	float total_Sd = 0.0;
	float total_Sr = 0.0;
	float total_StCS = 0.0;
	float ave_Sd = 0.0;
	float ave_Sr = 0.0;
	float ave_StCS = 0.0;

	// ���㵱ǰʱ�䲽���ϵ�ƽ���ݵ�ˮ�ƽ���ر�ˮ�ƽ��ʣ��洢����
	for (int i = 0; i < m_nCells; i++)
	{
		total_Sd += m_sd[i];
		total_Sr += m_sr[i];
		total_StCS += m_storageCapSurplus[i];
	}
	if (total_Sd < 0.00001)
	{
		total_Sd = 0.f;
	}
	ave_Sd = total_Sd / m_nCells;
	if (total_Sr < 0.00001)
	{
		total_Sr = 0.f;
	}
	ave_Sr = total_Sr / m_nCells;
	if (total_StCS < 0.00001)
	{
		total_StCS = 0.f;
	}
	ave_StCS = total_StCS / m_nCells;
	//cout << "average storage depression: " << ave_Sd << "mm" << endl;
	//cout << "average surface runoff: " << ave_Sr << "mm" << endl;
	//cout << "average storage surplus: " << ave_StCS << "mm" << endl;
    return 0;
}

bool DepressionFS::CheckInputSize(const char *key, int n) {
    if (n <= 0) {
        //StatusMsg("Input data for "+string(key) +" is invalid. The size could not be less than zero.");
        return false;
    }
    if (m_nCells != n) {
        if (m_nCells <= 0) { m_nCells = n; }
        else {
            //StatusMsg("Input data for "+string(key) +" is invalid. All the input data should have same size.");
            ostringstream oss;
            oss << "Input data for " + string(key) << " is invalid with size: " << n << ". The origin size is " <<
                m_nCells << ".\n";
            throw ModelException(MID_DEP_FS, "CheckInputSize", oss.str());
        }
    }
    return true;
}

void DepressionFS::SetValue(const char *key, float data) {
    string sk(key);
    if (StringMatch(sk, VAR_DEPREIN)) {
        m_depCo = data;
    } else {
        throw ModelException(MID_DEP_FS, "SetValue", "Parameter " + sk
            + " does not exist in current module. Please contact the module developer.");
    }
}

void DepressionFS::Set1DData(const char *key, int n, float *data) {
    //check the input data
    CheckInputSize(key, n);
    string sk(key);
    if (StringMatch(sk, VAR_DEPRESSION)) {
        m_depCap = data;
		for (int i = 0; i < m_nCells; i++)
		{
			if (m_depCap[i] < 0.00001)
			{
				m_depCap[i] = 0.f;
			}
		}
    }
	//else if (StringMatch(sk, VAR_PET)) {
	//       m_pet = data;
	//   }
	else {
        throw ModelException(MID_DEP_FS, "Set1DData", "Parameter " + sk
            + " does not exist in current module. Please contact the module developer.");
    }
}

void DepressionFS::Get1DData(const char *key, int *n, float **data) {
    InitialOutputs();
    string sk(key);
    *n = m_nCells;
    if (StringMatch(sk, VAR_DPST)) {
        *data = m_sd;
    } else if (StringMatch(sk, VAR_SURU)) {
        *data = m_sr;
    } else if (StringMatch(sk, VAR_STCAPSURPLUS)) {
        *data = m_storageCapSurplus;
    } else {
        throw ModelException(MID_DEP_FS, "Get1DData", "Output " + sk
            + " does not exist in current module. Please contact the module developer.");
    }
}

