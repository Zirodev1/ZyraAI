# FindSndFile.cmake
# Find the libsndfile library
#
# This module defines
# SNDFILE_INCLUDE_DIRS - where to find sndfile.hh
# SNDFILE_LIBRARIES - the libraries needed to use libsndfile
# SNDFILE_FOUND - system has libsndfile

find_path(SNDFILE_INCLUDE_DIR
    NAMES sndfile.hh
    PATHS
        /opt/homebrew/include
        /usr/local/include
        /usr/include
    PATH_SUFFIXES sndfile
)

find_library(SNDFILE_LIBRARY
    NAMES sndfile
    PATHS
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SndFile DEFAULT_MSG
    SNDFILE_LIBRARY
    SNDFILE_INCLUDE_DIR
)

mark_as_advanced(SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)

set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY})
set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
