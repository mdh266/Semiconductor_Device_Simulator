#ifndef _BOUNDARY_VALUES_H___
#define _BOUNDARY_VALUES_H___

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

using namespace dealii;


///////////////////////////////////////////////////////////////////////////////
// Carrier Boundary Functions
///////////////////////////////////////////////////////////////////////////////


// Neumann Boundary Condition Class
template<int dim>
class Carrier_NeumannBoundaryValues : public Function<dim>
{
public:
    Carrier_NeumannBoundaryValues() : Function<dim>(dim)
    {}

    virtual void	vector_value(const Point<dim> &p,
                                 Point<dim>  			&values) const;

    virtual void	vector_value_list(const std::vector<Point<dim> > 		&points,
                                      std::vector<Point<dim> > 			&value_list) const;

};


// These have return values of Point<dim> as opposed to Vector<double>
// as in step-8, since the normal vectors are Point<dim> and assembling the
// flux for the Neumann BC we would have Vector<double> * Point<dim>
// which is undefined in deal.II
//

template<int dim>
inline
void Carrier_NeumannBoundaryValues<dim>::vector_value(const Point<dim> &p,
        Point<dim>  	&values) const
{
    for(unsigned int i=0; i<dim; i++)
        values(i) = 0.0;//-2.0 * p[i];
}

template<int dim>
void Carrier_NeumannBoundaryValues<dim>::
vector_value_list(const std::vector<Point<dim>> &points,
                  std::vector<Point<dim> >	  	&value_list) const
{
    Assert(value_list.size() == points.size(),
           ExcDimensionMismatch(value_list.size(), points.size()));

    // Avoids using the virtual function somehow because vector_value
    // is inlined and because of the way you call it below?
    for(unsigned int p=0; p<points.size(); p++)
        Carrier_NeumannBoundaryValues<dim>::vector_value(points[p],
                value_list[p]);

}

///////////////////////////////////////////////////////////////////////////////
// Poisson Boundary Functions
///////////////////////////////////////////////////////////////////////////////
template <int dim>
class Built_In_Bias : public dealii::Function<dim>
{
public:
    Built_In_Bias() : dealii::Function<dim>()
    {}

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0 ) const;
};

template <int dim>
double Built_In_Bias<dim>::value(const dealii::Point<dim> &p,
                                 const unsigned int ) const
{
    // potential applied at y = 0, a positive potential is a foward bias,
    // a negative potential is a reverse bias
    if(p[1] == 0.0)
        return -20.0;
    else
        return 0;
//	double return_value = 0.0;
//	for(unsigned int i = 0; i < dim; i++)
//		return_value += 20.0*p[i];
//	return return_value;
}

template <int dim>
class Applied_Bias : public dealii::Function<dim>
{
public:
    Applied_Bias() : dealii::Function<dim>()
    {}

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0 ) const;
};

template <int dim>
double Applied_Bias<dim>::value(const dealii::Point<dim> &p,
                                const unsigned int ) const
{
    // potential applied at y = 0, a positive potential is a foward bias,
    // a negative potential is a reverse bias
    if(p[1] == 1.0)
        return 0.0;
    else
        return 0;
//	double return_value = 0.0;
//	for(unsigned int i = 0; i < dim; i++)
//		return_value += 20.0*p[i];
//	return return_value;
}



#endif
