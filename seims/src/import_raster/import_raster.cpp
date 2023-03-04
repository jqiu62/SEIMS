/*!
 * \brief Import raster data (full size data include nodata) to MongoDB as GridFS.
 * \change  17-07-02 lj - keep import raster to MongoDB if fails and the max. loop is set to 3.\n
 *          18-05-06 lj - Use CCGL, and reformat code style.\n
 */
#if (defined _DEBUG) && (defined _MSC_VER) && (defined VLD)
#include "vld.h"
#endif /* Run Visual Leak Detector during Debug */

#include "basic.h"
#include "db_mongoc.h"
#include "data_raster.h"

using namespace ccgl;
using namespace data_raster;
using namespace db_mongoc;

#ifndef IntRaster
#define IntRaster   clsRasterData<int>
#endif
#ifndef FloatRaster
#define FloatRaster clsRasterData<float>
#endif

struct SubBasin {
    SubBasin() : x_min(-1), y_min(-1), x_max(-1), y_max(-1), cell_count(0) {
    }

    SubBasin(const int xmin, const int ymin, const int xmax, const int ymax) {
        x_min = xmin;
        y_min = ymin;
        x_max = xmax;
        y_max = ymax;
        cell_count = 0;
    }

    ~SubBasin() {
    }

    int x_min;
    int y_min;
    int x_max;
    int y_max;
    int cell_count; ///< count of valid cells
};

int FindBoundingBox(IntRaster* rs_subbasin, map<int, SubBasin>& bbox_map) {
    int n_xsize = rs_subbasin->GetCols();
    int n_ysize = rs_subbasin->GetRows();

    int* p_data = rs_subbasin->GetRasterDataPointer();
    int nodata_value = rs_subbasin->GetNoDataValue();// NODATA_VALUE    (-9999.0f)
    // find bounding box for each subbasin
    for (int i = 0; i < n_ysize; i++) {
        for (int j = 0; j < n_xsize; j++) {
            int id = p_data[i * n_xsize + j];		// դ���id��ʶ
            if (nodata_value != id) {
                if (bbox_map.find(id) == bbox_map.end()) {	// ���û�ҵ�
                    bbox_map[id] = SubBasin(j, i, j, i);				// ����ǰid��������Χ����Ϊ(j, i, j, i)������ǰ��դ��Ԫ
                } else {
                    if (j < bbox_map[id].x_min) {						// Ѱ����С���к�x_min
                        bbox_map[id].x_min = j;
                    } else if (j > bbox_map[id].x_max) {				// Ѱ�������к�x_max
                        bbox_map[id].x_max = j;
                    }
                    if (i > bbox_map[id].y_max) {
                        bbox_map[id].y_max = i;
                    }
                }
                bbox_map[id].cell_count += 1;
            }
        }
    }
    return 0;
}

/*!
 * \brief Decomposite single layer raster data as 1D array, and import to MongoDB as GridFS
 * \param[in] bbox_map Map of subbasin extent box, key is subbasin id, value is SubBasin object
 * \param[in] rs_subbasin Subbasin raster object, clsRasterData
 * \param[in] dst_file Input raster full file path
 * \param[in] gfs MongoDB GridFS object
 * \return True if import successfully, otherwise return false.
 */
bool DecompositeRasterToMongoDB(map<int, SubBasin>& bbox_map, IntRaster* rs_subbasin,
                                const char* dst_file, mongoc_gridfs_t* gfs) {
    bool flag = true;
    FloatRaster* rs = FloatRaster::Init(dst_file, false); // Do not calculate positions data
    if (nullptr == rs) exit(1);
    int n_xsize = rs->GetCols();
    // int nYSize = rs.getRows();
    float nodata_value = rs->GetNoDataValue();
    //cout << nXSize << "\t" << nYSize << endl;
    float* rs_data = rs->GetRasterDataPointer();
    int* subbasin_data = rs_subbasin->GetRasterDataPointer();

    string core_name = GetCoreFileName(dst_file);

    for (auto it = bbox_map.begin(); it != bbox_map.end(); ++it) {	// ��������������
        int id = it->first;
        int subbasin_id = id;
        if (bbox_map.size() == 1) {
            id = 0;
        }
        SubBasin& subbasin = it->second;
        int sub_xsize = subbasin.x_max - subbasin.x_min + 1;
        int sub_ysize = subbasin.y_max - subbasin.y_min + 1;

        float* sub_data = new float[sub_xsize * sub_ysize];
#pragma omp parallel for
        for (int i = subbasin.y_min; i <= subbasin.y_max; i++) {
            /// row
            for (int j = subbasin.x_min; j <= subbasin.x_max; j++) {
                /// col
                int index = i * n_xsize + j;
                int sub_index = (i - subbasin.y_min) * sub_xsize + (j - subbasin.x_min);
                if (subbasin_data[index] == subbasin_id) {
                    sub_data[sub_index] = rs_data[index];
                } else {
                    sub_data[sub_index] = nodata_value;
                }
            }
        }

        std::ostringstream remote_filename;
        remote_filename << id << "_" << GetUpper(core_name);	// ����"0_SOL_SOLP"
        float cell_size = CVT_FLT(rs->GetCellWidth());
        MongoGridFs().RemoveFile(remote_filename.str(), gfs);

        bson_t p = BSON_INITIALIZER;
        BSON_APPEND_INT32(&p, "SUBBASIN", id);
        BSON_APPEND_UTF8(&p, "TYPE", core_name.c_str());
        BSON_APPEND_UTF8(&p, "ID", remote_filename.str().c_str());
        BSON_APPEND_UTF8(&p, "DESCRIPTION", core_name.c_str());
        BSON_APPEND_DOUBLE(&p, "CELLSIZE", cell_size);
        BSON_APPEND_DOUBLE(&p, "NODATA_VALUE", rs->GetNoDataValue());
        BSON_APPEND_DOUBLE(&p, "NCOLS", sub_xsize);
        BSON_APPEND_DOUBLE(&p, "NROWS", sub_ysize);
        BSON_APPEND_DOUBLE(&p, "XLLCENTER", rs->GetXllCenter() + subbasin.x_min * cell_size);
        BSON_APPEND_DOUBLE(&p, "YLLCENTER", rs->GetYllCenter() + (rs->GetRows() - subbasin.y_max - 1) * cell_size);
        BSON_APPEND_DOUBLE(&p, "LAYERS", rs->GetLayers());
        BSON_APPEND_DOUBLE(&p, "CELLSNUM", subbasin.cell_count);
        BSON_APPEND_UTF8(&p, "SRS", rs->GetSrsString().c_str());

        char* databuf = reinterpret_cast<char *>(sub_data);			//reinterpret_castǿ������ת��Ϊchar*
        size_t datalength = sizeof(float) * sub_xsize * sub_ysize;
        MongoGridFs().WriteStreamData(remote_filename.str(), databuf, datalength, &p, gfs);
        if (NULL == MongoGridFs().GetFile(remote_filename.str(), gfs)) {
            // import failed! Terminate the subbasins loop and return false.
            flag = false;
            break;
        }
        bson_destroy(&p);

        databuf = NULL;
        Release1DArray(sub_data);
    }
    delete rs;
    return flag;
}

/*!
 * \brief Decomposite multi-layers raster data as 1D array, and import to MongoDB as GridFS
 * \param[in] bbox_map Map of subbasin extent box, key is subbasin id, value is SubBasin object
 * \param[in] rs_subbasin Subbasin raster object, clsRasterData
 * \param[in] core_name Core name of raster
 * \param[in] dst_files Vector of raster full file paths
 * \param[in] conn MongoDB client object
 * \param[in] gfs MongoDB GridFS object
 * \return True if import successfully, otherwise return false.
 */
bool Decomposite2DRasterToMongoDB(map<int, SubBasin>& bbox_map, IntRaster* rs_subbasin,
                                  const string& core_name, vector<string> dst_files,
                                  mongoc_client_t* conn, mongoc_gridfs_t* gfs) {
    bool flag = true;
    int col_num = CVT_INT(dst_files.size());
    FloatRaster* rss = FloatRaster::Init(dst_files, false); // Do not calculate positions data
    if (nullptr == rss) exit(-1);
    int n_xsize = rss->GetCols();
    // int nYSize = rss.getRows();
    float nodata_value = rss->GetNoDataValue();
    ///cout << nXSize << "\t" << nYSize << endl;
    float** rss_data = rss->Get2DRasterDataPointer();
    int* subbasin_data = rs_subbasin->GetRasterDataPointer();
    for (auto it = bbox_map.begin(); it != bbox_map.end(); ++it) {	// ����ÿ��������
        int id = it->first;
        int subbasin_id = id;
        if (bbox_map.size() == 1) {
            id = 0;
        }
        SubBasin& subbasin = it->second;
        int sub_xsize = subbasin.x_max - subbasin.x_min + 1;
        int sub_ysize = subbasin.y_max - subbasin.y_min + 1;
        int sub_cell_num = sub_xsize * sub_ysize;
        float* sub_2ddata = nullptr;
        Initialize1DArray(sub_cell_num * col_num, sub_2ddata, nodata_value);
		cout << "������id: " << subbasin_id << endl;

#pragma omp parallel for
        for (int i = subbasin.y_min; i <= subbasin.y_max; i++) {
            for (int j = subbasin.x_min; j <= subbasin.x_max; j++) {		// �����������е�ÿ��դ��Ԫ
                int index = i * n_xsize + j;// �������е�դ��Ԫ����
                int sub_index = (i - subbasin.y_min) * sub_xsize + (j - subbasin.x_min);// �������е�դ��Ԫ����
				//cout << " i: " << i << " j: " << j << " index: " << index << " sub_index: " << sub_index << " subbasin_id: " << subbasin_id << " դ����Ԫֵ: " << subbasin_data[index] << endl;
				if (subbasin_data[index] == subbasin_id) {//�����ʾ������id��դ�����Ԫֵ����������id�� == ��ǰ�������id
					for (int k = 0; k < col_num; k++) {// ��k��ͼ��
                        sub_2ddata[sub_index * col_num + k] = rss_data[index][k];// sub_index * col_num + k������Ϊ����һά�����б���col_num��ͼ��
						//cout << "sub_index: " << sub_index <<" sub_2ddata[" << sub_index * col_num + k << "]: " << sub_2ddata[sub_index * col_num + k] << endl;
					}
                }
            }
        }
		//for (int i = 0; i < 1000; i++)
		//{
		//	cout << "sub_2ddata[" << i << "]: " << sub_2ddata[i];
		//}
        std::ostringstream remote_filename;
        remote_filename << id << "_" << GetUpper(core_name);
        float cell_size = CVT_FLT(rss->GetCellWidth());
        MongoGridFs().RemoveFile(remote_filename.str(), gfs);

        bson_t p = BSON_INITIALIZER;
        BSON_APPEND_INT32(&p, "SUBBASIN", id);
        BSON_APPEND_UTF8(&p, "TYPE", core_name.c_str());
        BSON_APPEND_UTF8(&p, "ID", remote_filename.str().c_str());
        BSON_APPEND_UTF8(&p, "DESCRIPTION", core_name.c_str());
        BSON_APPEND_DOUBLE(&p, "CELLSIZE", cell_size);
        BSON_APPEND_DOUBLE(&p, "NODATA_VALUE", rss->GetNoDataValue());
        BSON_APPEND_DOUBLE(&p, "NCOLS", sub_xsize);
        BSON_APPEND_DOUBLE(&p, "NROWS", sub_ysize);
        BSON_APPEND_DOUBLE(&p, "XLLCENTER", rss->GetXllCenter() + subbasin.x_min * cell_size);
        BSON_APPEND_DOUBLE(&p, "YLLCENTER", rss->GetYllCenter() + (rss->GetRows() - subbasin.y_max - 1) * cell_size);
        BSON_APPEND_DOUBLE(&p, "LAYERS", rss->GetLayers());
        BSON_APPEND_DOUBLE(&p, "CELLSNUM", subbasin.cell_count);
        BSON_APPEND_UTF8(&p, "SRS", rss->GetSrsString().c_str());

        char* databuf = reinterpret_cast<char *>(sub_2ddata);
        size_t datalength = sizeof(float) * sub_cell_num * col_num;
        MongoGridFs().WriteStreamData(remote_filename.str(), databuf, datalength, &p, gfs);
        if (NULL == MongoGridFs().GetFile(remote_filename.str(), gfs)) {
            // import failed! Terminate the subbasins loop and return false.
            flag = false;
            break;
        }
        bson_destroy(&p);
        databuf = nullptr;
        Release1DArray(sub_2ddata);
    }
    rss_data = nullptr;
    subbasin_data = nullptr;
    delete rss;
    return flag;
}

/*!
 * \brief Decomposite raster to separate files named suffixed by subbasin ID.
 *        If not for MPI version SEIMS, the whole basin data will be generated named suffixed by 0.
 * \param[in] bbox_map Map of subbasin extent box, key is subbasin id, value is SubBasin object
 * \param[in] rs_subbasin Subbasin raster object, clsRasterData
 * \param[in] dst_file Input raster full file path
 * \param[in] tmp_folder Folder to store the separated raster data file
 */
int DecompositeRaster(map<int, SubBasin>& bbox_map, IntRaster* rs_subbasin,
                      const char* dst_file, const char* tmp_folder) {	// dst_file���ļ�����ָ��
    FloatRaster* rs = FloatRaster::Init(dst_file, false);				// ��tif�ļ�����float���͵�դ��
    if (nullptr == rs) exit(-1);

    int n_xsize = rs->GetCols();
    float nodata_value = rs->GetNoDataValue();

    float* rs_data = rs->GetRasterDataPointer();
    int* subbasin_data = rs_subbasin->GetRasterDataPointer();

    const char* psz_format = "GTiff";
    GDALDriver* po_driver = GetGDALDriverManager()->GetDriverByName(psz_format);

    string core_name = GetCoreFileName(dst_file);
    for (auto it = bbox_map.begin(); it != bbox_map.end(); ++it) {
        int id = it->first;						// �� subbasin id ֵ subbasin����
        int subbasin_id = id;
        if (bbox_map.size() == 1) {
            // means not for Cluster
            id = 0;
        }
        SubBasin& subbasin = it->second;
        int subXSize = subbasin.x_max - subbasin.x_min + 1; //X����
        int subYSize = subbasin.y_max - subbasin.y_min + 1; //Y����

        std::ostringstream oss;
        oss << tmp_folder << "/" << id;
        string dst_folder = oss.str();
        if (!DirectoryExists(dst_folder)) {
            /// create new directory
#ifdef WINDOWS
            LPSECURITY_ATTRIBUTES att = nullptr;
            ::CreateDirectory(dst_folder.c_str(), att);
#else
            mkdir(dst_folder.c_str(), 0777);
#endif /* WINDOWS */
        }
        oss << "/" << GetUpper(core_name) << ".tif";
        string subbasin_file = oss.str();						// Ҫд���Ŀ���ļ�
        //cout << subbasinFile << endl;
        // create a new raster for the subbasin
        char** papsz_options = nullptr;
        GDALDataset* po_dst_ds = po_driver->Create(subbasin_file.c_str(), subXSize, subYSize,
                                                   1, GDT_Float32, papsz_options);

        float* sub_data = reinterpret_cast<float *>(CPLMalloc(sizeof(float) * subXSize * subYSize));
#pragma omp parallel for
        for (int i = subbasin.y_min; i <= subbasin.y_max; i++) {									// �Ѷ�άդ�����ݶ�ȡ��һά������
            for (int j = subbasin.x_min; j <= subbasin.x_max; j++) {
                int index = i * n_xsize + j;
                int sub_index = (i - subbasin.y_min) * subXSize + (j - subbasin.x_min);	// ���±� = �к� * ÿ�е����� + �к�
                if (subbasin_data[index] == subbasin_id) {
                    sub_data[sub_index] = rs_data[index];
                } else {
                    sub_data[sub_index] = nodata_value;
                }
            }
        }

        //write the data to new file
        GDALRasterBand* po_dst_band = po_dst_ds->GetRasterBand(1);
        po_dst_band->RasterIO(GF_Write, 0, 0, subXSize, subYSize,		// �������壺д���ݡ�0,0���Ͻ�x,y���ꡢsubXSize��subYSizeд��ͼ�����ݵĴ��ڴ�С
                              sub_data, subXSize, subYSize, GDT_Float32, 0, 0);// Ҫд�������ָ��,subXSize, subYSize�������Ĵ�С
        po_dst_band->SetNoDataValue(nodata_value);

        double geo_trans[6];
        float cell_size = CVT_FLT(rs->GetCellWidth());
        geo_trans[0] = rs->GetXllCenter() + (subbasin.x_min - 0.5f) * cell_size;
        geo_trans[1] = cell_size;
        geo_trans[2] = 0;
        geo_trans[3] = rs->GetYllCenter() + (rs->GetRows() - subbasin.y_min - 0.5f) * cell_size;
        geo_trans[4] = 0;
        geo_trans[5] = -cell_size;
        po_dst_ds->SetGeoTransform(geo_trans);

        CPLFree(sub_data);
        GDALClose(po_dst_ds);
    }
    delete rs;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        cout << "Usage: import_raster <MaskFile> <DataFolder> <modelName> <GridFSName> "
                "<hostIP> <port> [outputFolder]\n";
        exit(-1);
    }

    GDALAllRegister();
    /// set default OpenMP thread number to improve compute efficiency
    SetDefaultOpenMPThread();

    const char* subbasin_file = argv[1];	// spatial_raster/subbasin.tif
    const char* folder = argv[2];				// spatial_raster
    const char* model_name = argv[3];	// demo_youwuzhen30m_longterm_model
    const char* gridfs_name = argv[4];		// SPATIAL
    const char* hostname = argv[5];
    char* strend = nullptr;
    int port = strtol(argv[6], &strend, 10);
    const char* out_tif_folder = nullptr;

    if (argc >= 8) {
        out_tif_folder = argv[7];
    }
    vector<string> dst_files;
    FindFiles(folder, "*.tif", dst_files);									// ��workspace/spatial_raster�в�������tif�ļ�
    cout << "File number:" << dst_files.size() << endl;
    /// Identify Array1D and Array2D dstFiles, respectively
    vector<string> core_file_names;
    vector<string> array1d_files;
    map<string, vector<string> > array2d_files;
    map<string, vector<string> >::iterator array2d_iter;
    for (auto it = dst_files.begin(); it != dst_files.end(); ++it) {		//	����tif�ļ�
        string tmp_core_name = GetCoreFileName(*it);					// ��ȡtif�ļ������ƣ�����SOL_SOLP_1.tif�ͻ��ȡSOL_SOLP_1
        core_file_names.push_back(tmp_core_name);					//	tif�ļ����ƴ���core_file_names verctor��
        vector<string> tokens = SplitString(tmp_core_name, '_');	//	[SOL,SOLP,1]
        int length = CVT_INT(tokens.size());									// 3

        if (length <= 1) {																	// �����к�1�����¡�_������1D���ݣ�����2����_������2D����
            array1d_files.push_back(*it);											// �����1D�ļ����ͽ������1D��vector
        } else if (length >= 2) {
            /// there are more than one underscore exist
            string::size_type end = tmp_core_name.find_last_of('_');
            string core_var_name = tmp_core_name.substr(0, end);// ���һ����_��֮ǰ�Ĳ��ִ��������SOL_SOLP
            if (strtol(tokens[length - 1].c_str(), &strend, 10)) {			// ȡ�����е����һ�����֣�baseΪ10ʱ���Ϸ��ַ�Ϊ0,1...9��strend�д洢�����������Ĳ��֣����û�����������Ľ�����򷵻�0ֵ
                array2d_iter = array2d_files.find(core_var_name);
                if (array2d_iter != array2d_files.end()) {						// һ��������Ӧ���tif�ļ��������array2d_files ��key���ҵ��˱��������Ͱ�tif�ļ�����ָ�����map��vector
                    array2d_files[core_var_name].push_back(*it);
                } else {																			// ���򴴽���vector��������������ָ�����vector����vector����map
                    vector<string> tmp_file_name;
                    tmp_file_name.push_back(*it);
#ifdef HAS_VARIADIC_TEMPLATES
                    array2d_files.emplace(core_var_name, tmp_file_name);
#else
                    array2d_files.insert(make_pair(core_var_name, tmp_file_name));
#endif
                }
            } else {
                array1d_files.push_back(*it);
            }
        }
    }
    vector<string> del_var_names;
    for (array2d_iter = array2d_files.begin(); array2d_iter != array2d_files.end(); ++array2d_iter) {	// ����2D�ļ�
        if (array2d_iter->second.size() == 1) {								// (*it).second��õ�value������ĸ�2D��������ֻ��Ӧһ���ļ�
            array1d_files.push_back(array2d_iter->second.at(0));		// ����2D���ļ�����1D vector��
            del_var_names.push_back(array2d_iter->first);				// ��2D map������ñ�����
        } else {
            sort(array2d_iter->second.begin(), array2d_iter->second.end());	// �����2D�ļ�����������
        }
    }
    for (auto it = del_var_names.begin(); it != del_var_names.end(); ++it) {	// // ��2D map�����1D������
        array2d_files.erase(*it);
    }
    vector<string>(array1d_files).swap(array1d_files);					// �ͷŶ�����ڴ�ռ�

    //////////////////////////////////////////////////////////////////////////
    // read the subbasin file, and find the bounding box of each subbasin
    IntRaster* rs_subbasin = IntRaster::Init(subbasin_file, false);	// ��ȡsubbasin.tif
    if (nullptr == rs_subbasin) exit(-1);
    map<int, SubBasin> bbox_map;
    FindBoundingBox(rs_subbasin, bbox_map);

    //////////////////////////////////////////////////////////////////////////
    /// loop to process the destination files

    /// connect to MongoDB
    MongoClient* client = MongoClient::Init(hostname, port);
    if (nullptr == client) exit(-1);
    mongoc_client_t* conn = client->GetConn();
    mongoc_gridfs_t* gfs = client->GetGridFs(string(model_name), string(gridfs_name));

    cout << "Importing spatial data to MongoDB...\n";
	// ��2D Raster����mongodb
    //for (array2d_iter = array2d_files.begin(); array2d_iter != array2d_files.end(); ++array2d_iter) {	// ����2D�ļ�����2D�ļ�תΪ1D���飬����mongodb
    //    vector<string> tmp_file_names = array2d_iter->second;											
    //    for (auto it = tmp_file_names.begin(); it != tmp_file_names.end(); ++it) {				// ������������Ӧ���ļ�������
    //        cout << "\t" << *it << endl;
    //        if (nullptr != out_tif_folder) {																				// �������Ĳ�������<8����δָ�����tif��λ�ã�������һ��
    //            DecompositeRaster(bbox_map, rs_subbasin, it->c_str(), out_tif_folder);		//  it->c_str()��ָ��it��ָ��
    //        }
    //    }
    //    int loop = 1;
    //    while (loop < 3) {
    //        if (!Decomposite2DRasterToMongoDB(bbox_map, rs_subbasin, array2d_iter->first,// array2d_iter->first������
    //                                          tmp_file_names, conn, gfs)) {// tmp_file_names1����������Ӧ�Ķ���ļ���
    //            cout << "Import " << array2d_iter->first << " failed, try time: " << loop << endl;
    //            loop++;
    //        } else {
    //            break;
    //        }
    //    }
    //    if (loop == 3) {
    //        cout << "ERROR! Exceed the max. try times please check and rerun!" << endl;
    //        exit(EXIT_FAILURE);
    //    }
    //}
	// ��1D Raster����mongodb
    for (size_t i = 0; i < array1d_files.size(); ++i) {
        cout << "\t" << array1d_files[i] << endl;
        if (out_tif_folder != nullptr) {
            DecompositeRaster(bbox_map, rs_subbasin, array1d_files[i].c_str(), out_tif_folder);
        }
        int loop = 1;
        while (loop < 3) {// ����3��
            if (!DecompositeRasterToMongoDB(bbox_map, rs_subbasin, array1d_files[i].c_str(), gfs)) {
                cout << "Import " << array1d_files[i] << " failed, try time: " << loop << endl;
                loop++;
            } else {
                break;
            }
        }
        if (loop == 3) {
            cout << "ERROR! Exceed max. try times please check and rerun!" << endl;
            exit(EXIT_FAILURE);
        }
    }
    /// release
    mongoc_gridfs_destroy(gfs);
    delete client;
    delete rs_subbasin;
}
