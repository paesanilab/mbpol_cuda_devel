#ifndef UTILITY_H
#define UTILITY_H

#include <map>
#include <vector>
#include <string>
#include <cstddef>
#include <limits>

// A 2D-array type based on vector
typedef std::vector<std::vector<double> > matrix_by_vector_t;

void clearMemo(double** & data);

void clearMemo(std::vector<double**> & data);

void clearMemo(std::map<std::string, double**> & data);

bool init_mtx_in_mem(double** & ptr, size_t& rows, size_t& cols);


bool IsFloat( std::string& myString );

int read2DArrayfile(double** & data, size_t& rows, size_t& cols, const char* file, int titleline=0, int thredhold_col=0, double thredhold_max=std::numeric_limits<double>::max());

void transpose_mtx(double** & datrsc, double** & datdst, size_t& nrow_rsc, size_t& ncol_rsc);


int stringRemoveDelimiter(char delimiter, const char *string);

bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref);

int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref);

bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref, std::string & string_retval);

#endif
