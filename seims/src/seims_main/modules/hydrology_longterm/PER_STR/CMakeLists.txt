MESSAGE(STATUS "        Compiling hydrology_longterm_subdir: PER_STR...")
SET(MODNAME PER_STR)
PROJECT(SEIMS_MODULE_${MODNAME})
FILE(GLOB SRC_LIST *.cpp *.h)
ADD_LIBRARY(${MODNAME} SHARED ${SRC_LIST})
SET(LIBRARY_OUTPUT_PATH ${SEIMS_BINARY_OUTPUT_PATH})
TARGET_LINK_LIBRARIES(${MODNAME} module_setting)
### For LLVM-Clang installed by brew, add link library of OpenMP explicitly.
IF(CV_CLANG AND LLVM_VERSION_MAJOR)
    TARGET_LINK_LIBRARIES(${MODNAME} ${OpenMP_LIBRARY})
ENDIF()
INSTALL(TARGETS ${MODNAME}
        RUNTIME DESTINATION "${INSTALL_DIR}/bin"
        PUBLIC_HEADER DESTINATION "${INSTALL_DIR}/include"
        ARCHIVE DESTINATION "${INSTALL_DIR}/lib"
        LIBRARY DESTINATION "${INSTALL_DIR}/lib")
IF (MSVC OR XCODE)
    SET_PROPERTY(TARGET ${MODNAME} PROPERTY FOLDER "modules/hydrology_longterm")
ENDIF ()
