#include "DriftDiffusionPoisson.cpp"



int main()
{
    try {
        using namespace dealii;
        using namespace LDG_MX;

        deallog.depth_console(0);
        int degree   		= 1;

        ParameterHandler					prm;
        ParameterReader						param(prm);
        param.read_parameters("input_file.prm");

        DriftDiffusionPoisson<2> 	DeviceSimulation(degree,prm);

        DeviceSimulation.run();

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }



    return 0;
}
