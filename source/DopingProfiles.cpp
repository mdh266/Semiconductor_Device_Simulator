#ifndef _DOPPING_PROFILES_H__
#define _DOPPING_PROFILES_H__

#include <deal.II/base/function.h>

using namespace dealii;

// TODO: Overload vector_value so it doesnt use virtual one
//

//NOTE:  Initial conditions function involves initial conditions for
// 			 current and density, though current is initially set to zero

template<int dim>
class DonorDoping : public Function<dim>
{
public:
    DonorDoping() : Function<dim>(dim+1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component=0) const;
    /*
    		virtual void vector_value(const Point<dim> & p,
    															Vector<double>	 & value) const;
    */
};

template<int dim>
double DonorDoping<dim>::value(const Point<dim> & p,
                               const unsigned int component) const
{
    // dim+1 components
    if(component < dim)
    {
        // set the components of the current initially to zero
        return ZeroFunction<dim>(dim+1).value(p, component);
    }
    else //(component == dim)
    {
        // set the density intial conditions here
        if(p[1] > 0.5)
            return 1.0; //-p.square();
        else
            return 0;

//	return 10.0*p.square();
    }
}

template<int dim>
class AcceptorDoping : public Function<dim>
{
public:
    AcceptorDoping() : Function<dim>(dim+1)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component=0) const;
    /*
    		virtual void vector_value(const Point<dim> & p,
    															Vector<double>	 & value) const;
    */
};

template<int dim>
double AcceptorDoping<dim>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
    // dim+1 components
    if(component < dim)
    {
        // set the components of the current initially to zero
        return ZeroFunction<dim>(dim+1).value(p, component);
    }
    else //(component == dim)
    {
//		return 10.0*p.square();

        // set the density intial conditions here
        if(p[1] < 0.5)
            return 1.0;//-p.square();
        else
            return 0;

    }
}

#endif
