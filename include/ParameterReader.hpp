#ifndef _PARAMETERREADER_H__
#define _PARAMETERREADER_H__

#include<deal.II/base/parameter_handler.h>

namespace LDG_MX
{
using namespace dealii;

class ParameterReader
{
public:
    ParameterReader(ParameterHandler & param_handler);
    void read_parameters(const std::string);

private:
    void	declare_parameters();
    ParameterHandler & prm;
};
}

#endif
