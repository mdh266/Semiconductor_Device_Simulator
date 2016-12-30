#include "../include/ParameterReader.hpp"

namespace LDG_MX
{
using namespace dealii;

ParameterReader::
ParameterReader(ParameterHandler & param_handler)
    :
    prm(param_handler)
{}

void
ParameterReader::
read_parameters(const std::string parameter_file)
{
    declare_parameters();
    prm.read_input(parameter_file);
}

void
ParameterReader::
declare_parameters()
{
    prm.enter_subsection("computational");
    prm.declare_entry("global refinements", "4",
                      Patterns::Integer(1,10),
                      "number of global refinements");

    prm.declare_entry("local refinements", "0",
                      Patterns::Integer(0,10),
                      "number of local refinements in predefined critical areas");

    prm.declare_entry("time step size", "0.01",
                      Patterns::Double(0,1),
                      "scaled time step size");

    prm.declare_entry("time stamps", "100",
                      Patterns::Integer(0,1000),
                      "number of output files");

    prm.leave_subsection();

    prm.enter_subsection("physical");

    prm.declare_entry("illumination status", "Off",
                      Patterns::Anything(),
                      "On means that the cell is illuminated, off means its not.");

    prm.declare_entry("characteristic length", "1.0e-4",
                      Patterns::Double(0),
                      "the characteristic length scale [cm]");

    prm.declare_entry("characteristic density", "1.0e16",
                      Patterns::Double(0),
                      "the characteristic density scale [cm^{-3}]");

    prm.declare_entry("characteristic time", "1.0e-12",
                      Patterns::Double(0),
                      "the characteristic time scale [s]");

    prm.declare_entry("end time", "1",
                      Patterns::Double(0),
                      "physical end time (in terms of characteristic time)");

    prm.declare_entry("intrinsic density", "2.564e9",
                      Patterns::Double(0),
                      "the intrinsic density scale [cm^{-3}]");

    prm.declare_entry("material permittivity", "11.9",
                      Patterns::Double(0),
                      "semiconductor permittivity const []");

    prm.declare_entry("photon flux", "1.2e17",
                      Patterns::Double(0),
                      "intensity of light  [cm^{-2}s^{-1} ]");

    prm.declare_entry("absorption coefficient", "1.74974e5",
                      Patterns::Double(0),
                      "absoprtion coefficient averaged over all energies  [cm^{-1} ]");

    prm.leave_subsection();

    prm.enter_subsection("electrons");

    prm.declare_entry("mobility", "1350.0",
                      Patterns::Double(0),
                      "electron mobility [v/cm^{2}]");

    prm.declare_entry("recombination time", "5e-5",
                      Patterns::Double(0),
                      "Recombination rate/time of electrons [s]");

    prm.leave_subsection();

    prm.enter_subsection("holes");

    prm.declare_entry("mobility", "480.0",
                      Patterns::Double(0),
                      "hole mobility [v/cm^{2}]");

    prm.declare_entry("recombination time", "5e-5",
                      Patterns::Double(0),
                      "Recombination rate/time of holes [s]");

    prm.leave_subsection();
}

}


