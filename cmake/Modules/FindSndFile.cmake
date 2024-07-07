# FindSndFile.cmake
# This module finds the sndfile library and headers
find_path(SNDFILE_INCLUDE_DIR sndfile.h)
find_library(SNDFILE_LIBRARY NAMES sndfile)

if (SNDFILE_INCLUDE_DIR AND SNDFILE_LIBRARY)
    set(SNDFILE_FOUND TRUE)
    set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY})
    set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
else ()
    set(SNDFILE_FOUND FALSE)
endif ()

mark_as_advanced(SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)
