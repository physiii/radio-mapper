# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andy/code/librtlsdr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andy/code/librtlsdr/build

# Include any dependencies generated for this target.
include src/CMakeFiles/rtl_adsb.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/rtl_adsb.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/rtl_adsb.dir/flags.make

src/CMakeFiles/rtl_adsb.dir/rtl_adsb.c.o: src/CMakeFiles/rtl_adsb.dir/flags.make
src/CMakeFiles/rtl_adsb.dir/rtl_adsb.c.o: ../src/rtl_adsb.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/CMakeFiles/rtl_adsb.dir/rtl_adsb.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtl_adsb.dir/rtl_adsb.c.o -c /home/andy/code/librtlsdr/src/rtl_adsb.c

src/CMakeFiles/rtl_adsb.dir/rtl_adsb.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtl_adsb.dir/rtl_adsb.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/rtl_adsb.c > CMakeFiles/rtl_adsb.dir/rtl_adsb.c.i

src/CMakeFiles/rtl_adsb.dir/rtl_adsb.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtl_adsb.dir/rtl_adsb.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/rtl_adsb.c -o CMakeFiles/rtl_adsb.dir/rtl_adsb.c.s

# Object files for target rtl_adsb
rtl_adsb_OBJECTS = \
"CMakeFiles/rtl_adsb.dir/rtl_adsb.c.o"

# External object files for target rtl_adsb
rtl_adsb_EXTERNAL_OBJECTS =

src/rtl_adsb: src/CMakeFiles/rtl_adsb.dir/rtl_adsb.c.o
src/rtl_adsb: src/CMakeFiles/rtl_adsb.dir/build.make
src/rtl_adsb: src/librtlsdr.so.0.5git
src/rtl_adsb: src/libconvenience_static.a
src/rtl_adsb: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
src/rtl_adsb: src/CMakeFiles/rtl_adsb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable rtl_adsb"
	cd /home/andy/code/librtlsdr/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rtl_adsb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/rtl_adsb.dir/build: src/rtl_adsb

.PHONY : src/CMakeFiles/rtl_adsb.dir/build

src/CMakeFiles/rtl_adsb.dir/clean:
	cd /home/andy/code/librtlsdr/build/src && $(CMAKE_COMMAND) -P CMakeFiles/rtl_adsb.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/rtl_adsb.dir/clean

src/CMakeFiles/rtl_adsb.dir/depend:
	cd /home/andy/code/librtlsdr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andy/code/librtlsdr /home/andy/code/librtlsdr/src /home/andy/code/librtlsdr/build /home/andy/code/librtlsdr/build/src /home/andy/code/librtlsdr/build/src/CMakeFiles/rtl_adsb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/rtl_adsb.dir/depend

