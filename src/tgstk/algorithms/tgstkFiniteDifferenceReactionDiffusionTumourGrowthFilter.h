/*==========================================================================

  Program:   Tumour Growth Simulation Toolkit
  Module:    tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter.h

  Copyright (c) Corentin Martens
  All rights reserved.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
     NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
     ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR
     OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING
     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
     OTHER DEALINGS IN THE SOFTWARE.

==========================================================================*/

/**
 *
 * @class tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter
 *
 * @brief Finite difference solver for reaction-diffusion tumour growth
 * simulation over regular grids.
 *
 * tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter solves the
 * reaction-diffusion tumour growth problem introduced in
 * \cite konukoglu_2007 \cite hogea_2008 \cite swanson_2008
 * \cite rockne_2010 \cite unkelbach_2014 :
 *
 * \f{cases}
 * &\frac{\partial c(\bar{r}, t)}{\partial t} = \bar{\nabla} \cdot \left(
 * \bar{\bar{D}}(\bar{r}) \, \bar{\nabla} c(\bar{r}, t) \right) +
 * \rho(\bar{r}) \, c(\bar{r}, t) \left( 1-c(\bar{r}, t) \right) & \forall
 * \bar{r} \in \Omega, \; \forall t > 0 \\
 * &c(\bar{r}, 0) = c_0(\bar{r}) & \forall \bar{r} \in \Omega \\
 * &\left( \bar{\bar{D}}(\bar{r}) \, \bar{\nabla} c(\bar{r}, t) \right)
 * \cdot \bar{n}_{\partial \Omega}(\bar{r}) = 0 & \forall \bar{r} \in
 * \partial \Omega
 * \f}
 *
 * where \f$c(\bar{r}, t)\f$ is the nomarlised tumour cell density at
 * location \f$\bar{r}\f$ and time \f$t\f$ with \f$c(\bar{r}, t) \in
 * [0, 1], \; \forall \bar{r}, t\f$; \f$\bar{\bar{D}}(\bar{r})\f$ is the
 * tumour diffusion tensor field; \f$\rho(\bar{r})\f$ is the tumour
 * proliferation rate field; \f$\Omega\f$ is the solving domain;
 * \f$c_0(\bar{r})\f$ is the initial normalised cell density field at time
 * \f$t=0\f$; and \f$\bar{n}_{\partial \Omega}(\bar{r})\f$ is the unit
 * normal vector pointing outwards the domain boundary \f$\partial \Omega\f$
 * at location \f$\bar{r} \in \partial \Omega\f$.
 *
 * The problem is solved using a forward Euler finite difference scheme
 * implemented in CUDA:
 *
 * \f[
 * c_{i,j,k}^{n+1} = c_{i,j,k}^n + \Delta t \left(\text{div}_{i,j,k}^n +
 * \rho_{i,j,k} \, c_{i,j,k}^n \, (1-c_{i,j,k}^n) \right)
 * \f]
 *
 * where subscript \f$i, j, k\f$ refers to voxel \f$(i,j,k)\f$; superscript
 * \f$n\f$ refers to simulation step \f$n\f$; \f$\Delta t\f$ is the time
 * step; and \f$\text{div}_{i,j,k}^n\f$ is the divergence of the tumour cell
 * flux given by \f$\bar{\nabla} \cdot \left(\bar{\bar{D}}(\bar{r}) \,
 * \bar{\nabla} c(\bar{r}, t) \right)\f$. The divergence term
 * \f$\text{div}_{i,j,k}^n\f$ is computed using a 3D extension of the
 * standard stencil presented in \cite mosayebi_2010.
 *
 * The initial tumour cell density \f$c_0(\bar{r})\f$, diffusion tensor
 * \f$\bar{\bar{D}}(\bar{r})\f$, and proliferation rate \f$\rho(\bar{r})\f$
 * fields as well as the brain map defining the domain \f$\Omega\f$ can be
 * specified by the user as
 * <a href=https://vtk.org/doc/nightly/html/classvtkImageData.html>
 * vtkImageData</a> objects. The simulated time \f$T\f$ and time step
 * \f$\Delta t\f$ can also be specified.
 *
 * @warning As the CUDA implementation performs in single precision floating
 * point, a loss of precision may occur.
 *
 */

#ifndef TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONTUMOURGROWTHFILTER_H
#define TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONTUMOURGROWTHFILTER_H

#include <tgstk/algorithms/tgstkImageProcessorBase.h>

class TGSTK_EXPORT tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter : public virtual tgstkImageProcessorBase {

    public:

        tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter();
        ~tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter();

        bool check();
        void execute();

        /**
         *
         * Gets the final normalised tumour cell density image
         * \f$c(\bar{r}, T)\f$.
         *
         * The image scalar type is VTK_DOUBLE.
         *
         */
        vtkSmartPointer<vtkImageData> getFinalCellDensityImage();

        /**
         *
         * Gets the final normalised tumour cell density gradient image
         * \f$\bar{\nabla}c(\bar{r}, T)\f$ in \f$\text{mm}^{-1}\f$.
         *
         * The image scalar type is VTK_DOUBLE. The image has 3
         * (\f$\!\partial_x c, \partial_y c, \partial_z c\f$) scalar
         * components.
         *
         */
        vtkSmartPointer<vtkImageData> getFinalCellDensityGradientImage();


        /**
         *
         * Sets the bain map image defining the solving domain \f$\Omega\f$.
         *
         * The image scalar type must be VTK_UNSIGNED_SHORT. The different
         * brain tissues must be referenced as specified by the
         * ::tgstkBrainTissueType enum type.
         *
         */
        void setBrainMapImage(vtkImageData* image);

        /**
         *
         * Sets the tumour diffusion tensor image \f$\bar{\bar{D}}
         * (\bar{r})\f$ in \f$\text{mm}^2\text{ d}^{-1}\f$.
         *
         * The image scalar type must be VTK_DOUBLE. The image must have 6
         * scalar components (\f$\!D_{xx}, D_{xy},  D_{xz}, D_{yy}, D_{yz},
         * D_{zz}\f$ -- implicitly symmetric).
         *
         */
        void setDiffusionTensorImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the initial normalised tumour cell density image
         * \f$c_0(\bar{r})\f$.
         *
         * The image scalar type must be VTK_DOUBLE. The image values must
         * be in range \f$[0; 1]\f$.
         *
         */
        void setInitialCellDensityImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the tumour proliferation rate image \f$\rho(\bar{r})\f$ in
         * \f$\text{d}^{-1}\f$.
         *
         * The image scalar type must be VTK_DOUBLE. The image values must
         * be \f$\geq 0\f$.
         *
         */
        void setProliferationRateImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the time duration to simulate \f$T\f$ in \f$\text{d}\f$.
         *
         * The value must be \f$\geq 0\f$.
         *
         */
        void setSimulatedTime(double time);

        /**
         *
         * Sets the smilation time step \f$\Delta t\f$ in \f$\text{d}\f$.
         *
         * The value must be \f$> 0\f$.
         *
         * @warning For numerical stability, the time step value should
         * verify:
         * \f[
         * \Delta t \leq \min_{\bar{r}} \frac{1}{2} \left(\frac{D_{xx}
         * (\bar{r})}{\Delta x^2} + \frac{D_{yy}(\bar{r})}{\Delta y^2} +
         * \frac{D_{zz}(\bar{r})}{\Delta z^2}\right)^{-1}
         * \f]
         * where \f$\Delta x\f$, \f$\Delta y\f$, and \f$\Delta z\f$ are
         * respectively the voxel size in \f$x\f$, \f$y\f$, and \f$z\f$ in
         * \f$\text{mm}\f$.
         *
         */
        void setTimeStep(double step);

        //void setCoreProliferationRate(double proliferationRate);
        //void setNonCoreProliferationRate(double proliferationRate);

    private:

        //double coreProliferationRate;
        //double nonCoreProliferationRate;
        double simulatedTime;
        double timeStep;

        vtkSmartPointer<vtkImageData> brainMapImage;
        vtkSmartPointer<vtkImageData> finalCellDensityImage;
        vtkSmartPointer<vtkImageData> finalCellDensityGradientImage;
        vtkSmartPointer<vtkImageData> initialCellDensityImage;
        vtkSmartPointer<vtkImageData> proliferationRateImage;
        vtkSmartPointer<vtkImageData> diffusionTensorImage;
};

#endif // TGSTKFINITEDIFFERENCEREACTIONDIFFUSIONTUMOURGROWTHFILTER_H
