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
include src/CMakeFiles/rtlsdr_static.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/rtlsdr_static.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/rtlsdr_static.dir/flags.make

src/CMakeFiles/rtlsdr_static.dir/librtlsdr.c.o: src/CMakeFiles/rtlsdr_static.dir/flags.make
src/CMakeFiles/rtlsdr_static.dir/librtlsdr.c.o: ../src/librtlsdr.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/CMakeFiles/rtlsdr_static.dir/librtlsdr.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtlsdr_static.dir/librtlsdr.c.o -c /home/andy/code/librtlsdr/src/librtlsdr.c

src/CMakeFiles/rtlsdr_static.dir/librtlsdr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtlsdr_static.dir/librtlsdr.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/librtlsdr.c > CMakeFiles/rtlsdr_static.dir/librtlsdr.c.i

src/CMakeFiles/rtlsdr_static.dir/librtlsdr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtlsdr_static.dir/librtlsdr.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/librtlsdr.c -o CMakeFiles/rtlsdr_static.dir/librtlsdr.c.s

src/CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.o: src/CMakeFiles/rtlsdr_static.dir/flags.make
src/CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.o: ../src/tuner_e4k.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.o -c /home/andy/code/librtlsdr/src/tuner_e4k.c

src/CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/tuner_e4k.c > CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.i

src/CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/tuner_e4k.c -o CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.s

src/CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.o: src/CMakeFiles/rtlsdr_static.dir/flags.make
src/CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.o: ../src/tuner_fc0012.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object src/CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.o -c /home/andy/code/librtlsdr/src/tuner_fc0012.c

src/CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/tuner_fc0012.c > CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.i

src/CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/tuner_fc0012.c -o CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.s

src/CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.o: src/CMakeFiles/rtlsdr_static.dir/flags.make
src/CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.o: ../src/tuner_fc0013.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object src/CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.o -c /home/andy/code/librtlsdr/src/tuner_fc0013.c

src/CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/tuner_fc0013.c > CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.i

src/CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/tuner_fc0013.c -o CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.s

src/CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.o: src/CMakeFiles/rtlsdr_static.dir/flags.make
src/CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.o: ../src/tuner_fc2580.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object src/CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.o -c /home/andy/code/librtlsdr/src/tuner_fc2580.c

src/CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/tuner_fc2580.c > CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.i

src/CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/tuner_fc2580.c -o CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.s

src/CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.o: src/CMakeFiles/rtlsdr_static.dir/flags.make
src/CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.o: ../src/tuner_r82xx.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object src/CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.o"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.o -c /home/andy/code/librtlsdr/src/tuner_r82xx.c

src/CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.i"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/andy/code/librtlsdr/src/tuner_r82xx.c > CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.i

src/CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.s"
	cd /home/andy/code/librtlsdr/build/src && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/andy/code/librtlsdr/src/tuner_r82xx.c -o CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.s

# Object files for target rtlsdr_static
rtlsdr_static_OBJECTS = \
"CMakeFiles/rtlsdr_static.dir/librtlsdr.c.o" \
"CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.o" \
"CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.o" \
"CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.o" \
"CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.o" \
"CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.o"

# External object files for target rtlsdr_static
rtlsdr_static_EXTERNAL_OBJECTS =

src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/librtlsdr.c.o
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/tuner_e4k.c.o
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/tuner_fc0012.c.o
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/tuner_fc0013.c.o
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/tuner_fc2580.c.o
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/tuner_r82xx.c.o
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/build.make
src/librtlsdr.a: src/CMakeFiles/rtlsdr_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/andy/code/librtlsdr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking C static library librtlsdr.a"
	cd /home/andy/code/librtlsdr/build/src && $(CMAKE_COMMAND) -P CMakeFiles/rtlsdr_static.dir/cmake_clean_target.cmake
	cd /home/andy/code/librtlsdr/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rtlsdr_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/rtlsdr_static.dir/build: src/librtlsdr.a

.PHONY : src/CMakeFiles/rtlsdr_static.dir/build

src/CMakeFiles/rtlsdr_static.dir/clean:
	cd /home/andy/code/librtlsdr/build/src && $(CMAKE_COMMAND) -P CMakeFiles/rtlsdr_static.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/rtlsdr_static.dir/clean

src/CMakeFiles/rtlsdr_static.dir/depend:
	cd /home/andy/code/librtlsdr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andy/code/librtlsdr /home/andy/code/librtlsdr/src /home/andy/code/librtlsdr/build /home/andy/code/librtlsdr/build/src /home/andy/code/librtlsdr/build/src/CMakeFiles/rtlsdr_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/rtlsdr_static.dir/depend

