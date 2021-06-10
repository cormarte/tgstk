/*==========================================================================

  This file is part of the Tumor Growth Simulation ToolKit (TGSTK)
  (<https://github.com/cormarte/TGSTK>, <https://cormarte.github.io/TGSTK>).

  Copyright (C) 2021  Corentin Martens

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <https://www.gnu.org/licenses/>.

  Contact: corentin.martens@ulb.be

==========================================================================*/

/**
 *
 * @class tgstkWaterToTumourCellDiffusionTensorImageFilter
 *
 * @brief Tumour cell diffusion tensor field computation from a DTI-derived
 * water diffusion tensor field.
 *
 * tgstkWaterToTumourCellDiffusionTensorImageFilter computes a tumour cell
 * diffusion tensor field \f$\bar{\bar{D}}(\bar{r})\f$ from the water
 * diffusion tensor field \f$\bar{\bar{D}}_\text{water}(\bar{r})\f$ assessed
 * by DTI as follows:
 *
 * \f[
 * \bar{\bar{D}}(\bar{r}) = \begin{cases}\bar{\bar{D}}_\text{white}(\bar{r})
 * \quad &\text{if } \bar{r} \in \Omega_\text{white} \bigcup
 * \Omega_\text{oedema} \bigcup \Omega_\text{enhancing} \bigcup
 * \Omega_\text{necrotic} \\ \bar{\bar{D}}_\text{grey} \quad &\text{if }
 * \bar{r} \in \Omega_\text{grey} \\ \bar{\bar{0}} \quad &\text{otherwise}
 * \end{cases}
 * \f]
 *
 * where \f$\bar{\bar{D}}_\text{white}(\bar{r})\f$ is the tumour diffusion
 * tensor field in white matter; \f$\bar{\bar{D}}_\text{grey}\f$ is the
 * constant tumour diffusion tensor in grey matter; \f$\bar{\bar{0}}\f$ is
 * the null tensor; and \f$\Omega_\text{white}\f$,
 * \f$\Omega_\text{oedema}\f$, \f$\Omega_\text{enhancing}\f$,
 * \f$\Omega_\text{necrotic}\f$, and \f$\Omega_\text{grey}\f$ are
 * respectively the white matter, oedema, enhancing core, necrotic core, and
 * grey matter subdomains.
 *
 * The anisotropic tumour diffusion tensor field in white matter
 * \f$\bar{\bar{D}}_\text{white}\f$ is built as proposed in
 * \cite jbabdi_2005 :
 *
 * \f[
 * \bar{\bar{D}}_\text{white} = \frac{3 \, d_\text{white}}
 * {\sum_{i=1}^3 \tilde{\lambda}_i(a)} \sum_{i=1}^3 \tilde{\lambda}_i(a) \,
 * \bar{e}_i \, \bar{e}_i^\top
 * \f]
 *
 * where \f$d_\text{white}\f$ is the tumour cell mean diffusivity in
 * white matter; \f$\bar{e}_i\f$ is the \f$i^\text{th}\f$ eigenvector of the
 * water diffusion tensor \f$\bar{\bar{D}}_\text{water}\f$; and
 * \f$\tilde{\lambda}_i(a) = l_i(a)\lambda_i\f$ with:
 *
 * \f[
 * \begin{bmatrix}
 * l_1(a) \\ l_2(a) \\ l_3(a) \end{bmatrix} = \begin{bmatrix} a & a & 1 \\
 * 1 & a & 1 \\ 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} c_l \\ c_p \\ c_s
 * \end{bmatrix}
 * \f]
 *
 * where \f$a\f$ is the anisotropy factor and:
 *
 * \f[
 * c_l = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}, \quad
 * c_p = \frac{2 \, (\lambda_2-\lambda_3)}{\lambda_1+\lambda_2+\lambda_3},
 * \quad c_s = \frac{3 \, \lambda_3}{\lambda_1+\lambda_2+\lambda_3}
 * \f]
 *
 * \f$\lambda_i\f$ being the \f$i^\text{th}\f$ eigenvalue of the water
 * diffusion tensor \f$\bar{\bar{D}}_\text{water}\f$.
 *
 * The isotropic constant tumour diffusion tensor in grey matter
 * \f$\bar{\bar{D}}_\text{grey}\f$ is given by:
 *
 * \f[
 * \bar{\bar{D}}_\text{grey} = \begin{bmatrix} d_\text{grey} & 0 & 0 \\
 * 0 & d_\text{grey} & 0 \\ 0 & 0 & d_\text{grey} \end{bmatrix}
 * \f]
 *
 * where \f$d_\text{grey}\f$ is the tumour cell mean diffusivity in grey
 * matter.
 *
 * The water diffusion tensor field \f$\bar{\bar{D}}_\text{water}
 * (\bar{r})\f$ and the brain map defining the domain \f$\Omega\f$ can be
 * specified by the user as
 * <a href=https://vtk.org/doc/nightly/html/classvtkImageData.html>
 * vtkImageData</a> objects. The white and grey matter tumour cell mean
 * diffusivites \f$d_\text{white}\f$ and \f$d_\text{grey}\f$ can also be
 * specified. Alternatively, an isotropic constant tumour diffusion tensor
 * can be used for white matter as well.
 *
 */

#ifndef TGSTKWATERTOCELLDIFFUSIONTENSORIMAGEFILTER_H
#define TGSTKWATERTOCELLDIFFUSIONTENSORIMAGEFILTER_H

#include <tgstk/algorithms/tgstkImageProcessorBase.h>

class TGSTK_EXPORT tgstkWaterToTumourCellDiffusionTensorImageFilter : public virtual tgstkImageProcessorBase {

    public:

        enum TensorMode {

            ANISOTROPIC,
            ISOTROPIC
        };

        tgstkWaterToTumourCellDiffusionTensorImageFilter();
        ~tgstkWaterToTumourCellDiffusionTensorImageFilter();

        bool check();
        void execute();

        /**
         *
         * Gets the tumour cell diffusion tensor image \f$\bar{\bar{D}}
         * (\bar{r})\f$ in \f$\text{mm}^2\text{ d}^{-1}\f$.
         *
         * The image scalar type is VTK_DOUBLE. The image has 6 scalar
         * components (\f$\!D_{xx}, D_{xy},  D_{xz}, D_{yy}, D_{yz}, D_{zz}
         * \f$).
         *
         */
        vtkSmartPointer<vtkImageData> getTumourCellDiffusionTensorImage();

        /**
         *
         * Sets the anisotropy factor \f$a\f$.
         *
         * The value must be \f$\geq 1\f$.
         *
         */
        void setAnisotropyFactor(double factor);

        /**
         *
         * Sets the bain map image defining the solving domain \f$\Omega\f$.
         *
         * The image scalar type must be VTK_UNSIGNED_SHORT. The different
         * brain tissues must be referenced as specified by the
         * ::tgstkBrainTissueType enum type.
         *
         */
        void setBrainMapImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the grey matter mean diffusivity \f$d_\text{grey}\f$ in
         * \f$\text{mm}^2\text{ d}^{-1}\f$.
         *
         * The value must be \f$\geq 0\f$.
         *
         */
        void setGreyMatterMeanDiffusivity(double diffusivity);

        /**
         *
         * Sets the diffusion tensor mode.
         *
         * tgstkWaterToTumourCellDiffusionTensorImageFilter::ANISOTROPIC
         * will produce an anisotropic tumour cell diffusion tensor field in
         * white matter \f$\bar{\bar{D}}_\text{white}(\bar{r})\f$ derived
         * from a user-provided water diffusion tensor field
         * \f$\bar{\bar{D}}_\text{water}(\bar{r})\f$ assessed by DTI using
         * the method proposed in \cite jbabdi_2005.
         * tgstkWaterToTumourCellDiffusionTensorImageFilter::ISOTROPIC
         * will produce a constant isotropic tumour cell diffusion tensor
         * in white matter of mean diffusivity  \f$d_\text{white}\f$. No
         * water diffusion tensor field must be provided for this mode.
         *
         */
        void setTensorMode(TensorMode mode);

        /**
         *
         * Sets the diffusion tensor mode to
         * tgstkWaterToTumourCellDiffusionTensorImageFilter::ANISOTROPIC.
         *
         * Will produce an anisotropic tumour cell diffusion tensor field in
         * white matter \f$\bar{\bar{D}}_\text{white}(\bar{r})\f$ derived
         * from a user-provided water diffusion tensor field
         * \f$\bar{\bar{D}}_\text{water}(\bar{r})\f$ assessed by DTI using
         * the method proposed in \cite jbabdi_2005.
         *
         */
        void setTensorModeToAnisotropic();

        /**
         *
         * Sets the diffusion tensor mode to
         * tgstkWaterToTumourCellDiffusionTensorImageFilter::ISOTROPIC.
         *
         * Will produce a constant isotropic tumour cell diffusion tensor
         * in white matter of mean diffusivity  \f$d_\text{white}\f$. No
         * water diffusion tensor field must be provided for this mode.
         *
         */
        void setTensorModeToIsotropic();

        /**
         *
         * Sets the water diffusion tensor image
         * \f$\bar{\bar{D}}_\text{water}(\bar{r})\f$ assessed by DTI in
         * \f$\text{mm}^2\text{ d}^{-1}\f$.
         *
         * The image scalar type must be VTK_DOUBLE. The image must have 6
         * (\f$\!D_{xx}, D_{xy},  D_{xz}, D_{yy}, D_{yz}, D_{zz}\f$ --
         * implicitly symmetric) or 9 (\f$\!D_{xx}, D_{xy},  D_{xz}, D_{yx},
         * D_{yy}, D_{yz}, D_{zx}, D_{zy}, D_{zz}\f$) scalar components.
         * This image is required and used only if
         * tgstkWaterToTumourCellDiffusionTensorImageFilter::tensorMode is
         * tgstkWaterToTumourCellDiffusionTensorImageFilter::ANISOTROPIC.
         *
         */
        void setWaterDiffusionTensorImage(vtkSmartPointer<vtkImageData> image);

        /**
         *
         * Sets the white matter mean diffusivity \f$d_\text{white}\f$ in
         * \f$\text{mm}^2\text{ d}^{-1}\f$.
         *
         * The value must be \f$\geq 0\f$.
         *
         */
        void setWhiteMatterMeanDiffusivity(double diffusivity);

    private:

        double anisotropyFactor;
        TensorMode tensorMode;
        double greyMatterMeanDiffusivity;
        double whiteMatterMeanDiffusivity;

        vtkSmartPointer<vtkImageData> brainMapImage;
        vtkSmartPointer<vtkImageData> tumourCellDiffusionTensorImage;
        vtkSmartPointer<vtkImageData> waterDiffusionTensorImage;
};

#endif // TGSTKWATERTOCELLDIFFUSIONTENSORIMAGEFILTER_H
