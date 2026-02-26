# CMake generated Testfile for 
# Source directory: C:/Users/Magewell/Desktop/EE/miniDSP_library
# Build directory: C:/Users/Magewell/Desktop/EE/miniDSP_library/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_dsp_lib "C:/Users/Magewell/Desktop/EE/miniDSP_library/build/test_dsp_lib.exe")
set_tests_properties(test_dsp_lib PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Magewell/Desktop/EE/miniDSP_library/CMakeLists.txt;22;add_test;C:/Users/Magewell/Desktop/EE/miniDSP_library/CMakeLists.txt;0;")
subdirs("googletest")
