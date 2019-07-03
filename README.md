# Compile as:

g++ -pg Hungarian.cpp -o heuristic -std=c++11 heuristic.cpp -I /usr/local/include/eigen3 -lboost_iostreams -lboost_system -lboost_filesystem  

# Run as:

./heuristic M N -read

M -> no of vehicles
N -> no of tasks
-read or -random -> read given delta matrices or generate randomly
