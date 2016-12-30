#ifndef _TEST_FUNCTIONS_H__
#define _TEST_FUNCTIONS_H__

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>

namespace test_Poisson
{

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide() : Function<dim>(1)
    {}

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0 ) const;
};

template <int dim>
class DirichletBoundaryValues : public Function<dim>
{
public:
    DirichletBoundaryValues() : Function<dim>(1)
    {}

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0 ) const;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int ) const
{
    double x = p[0];
    double z = x*x - 1;
    double y = p[1];

//		double return_value = 0.0;
//		for(unsigned int i=0; i<dim; i++)
    //		return_value += -6.0 * p[i];

    return -6*z*y*(5*x*x-1);
}

template <int dim>
double DirichletBoundaryValues<dim>::value(const Point<dim> &p,
        const unsigned int ) const
{

    /*		double return_value;
    		for(unsigned int d=0; d<dim; d++)
    			return_value += p[d]*p[d]*p[d];

    		return return_value;
    */
    double x = p[0];
    double z = x*x - 1;
    double y = p[1];

    return z*z*z*y;

}


//////////////////////////////////////////////////////////////////////////////
// TEST CASE
///////////////////////////////////////////////////////////////////////////////
template<int dim>
class TrueSolution : public Function<dim>
{
public:
    TrueSolution() : Function<dim>(dim+1)
    {}

    virtual void vector_value(const Point<dim> & p,
                              Vector<double> &valuess) const;
    /*
    	virtual double value(const Point<dim> &p,
    											 const unsigned int component = 0 ) const;
    */
};

template <int dim>
void TrueSolution<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const
{
    Assert(values.size() == dim+1,
           ExcDimensionMismatch(values.size(), dim+1) );

    double x = p[0];
    double y = p[1];
    double z = (x*x - 1);

    values(0) = -6*x*z*z*y;
    values[1] = -z*z*z;
    values[2] =	z*z*z*y;

    /*
    		double pot_value = 0.0;
    		for(unsigned int i = 0; i < dim; i++)
    		{
    			values(i) = -3 *p(i) * p(i);
    			pot_value += p(i) * p(i) * p(i);
    		}
    		values(dim) = pot_value;
    */
}
/*
	// returns the dirichlet value for the potential
	template <int dim>
	double TrueSolution<dim>::value(const Point<dim> &p,
																	const unsigned int ) const
	{

	double x = p[0];
	double z = x*x - 1;
	double y = p[1];

	return z*z*z*y;

	}
*/

}
#endif
