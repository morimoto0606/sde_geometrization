# CMake generated Testfile for 
# Source directory: /home/morimoto/Public/GitHub/QuantsCpp/sde_geometrization
# Build directory: /home/morimoto/Public/GitHub/QuantsCpp/sde_geometrization/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MyTest "/home/morimoto/Public/GitHub/QuantsCpp/sde_geometrization/build/test/Test")
set_tests_properties(MyTest PROPERTIES  _BACKTRACE_TRIPLES "/home/morimoto/Public/GitHub/QuantsCpp/sde_geometrization/CMakeLists.txt;7;add_test;/home/morimoto/Public/GitHub/QuantsCpp/sde_geometrization/CMakeLists.txt;0;")
subdirs("src")
subdirs("test")
