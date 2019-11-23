include_directories(BEFORE ..)

add_executable(fuser main.cc)

target_link_libraries(fuser PRIVATE ethcore poolprotocols devcore ethminer-buildinfo Boost::system Boost::thread apicore)
