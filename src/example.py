import matplotlib.pyplot as plt
import numpy as np
import sys
import vtk
from os.path import join
from tgstk import tgstk
from vtk_numpy_utils import image_to_array


def run_example(data_dir):

    # Image reading
    brain_map_reader = vtk.vtkMetaImageReader()
    brain_map_reader.SetFileName(join(data_dir, 'domain.mha'))
    brain_map_reader.Update()
    brain_map_image = brain_map_reader.GetOutput()

    diffusion_tensor_reader = vtk.vtkMetaImageReader()
    diffusion_tensor_reader.SetFileName(join(data_dir, 'tensor.mha'))
    diffusion_tensor_reader.Update()
    diffusion_tensor_image = diffusion_tensor_reader.GetOutput()

    proliferation_rate_reader = vtk.vtkMetaImageReader()
    proliferation_rate_reader.SetFileName(join(data_dir, 'proliferation.mha'))
    proliferation_rate_reader.Update()
    proliferation_rate_image = proliferation_rate_reader.GetOutput()

    initial_cell_density_reader = vtk.vtkMetaImageReader()
    initial_cell_density_reader.SetFileName(join(data_dir, 'initial.mha'))
    initial_cell_density_reader.Update()
    initial_cell_density_image = initial_cell_density_reader.GetOutput()

    # Tumour growth simulation
    filter = tgstk.tgstkFiniteDifferenceReactionDiffusionTumourGrowthFilter()
    filter.setBrainMapImage(brain_map_image)
    filter.setDiffusionTensorImage(diffusion_tensor_image)
    filter.setInitialCellDensityImage(initial_cell_density_image)
    filter.setProliferationRateImage(proliferation_rate_image)
    filter.setSimulatedTime(120.0)
    filter.setTimeStep(0.05)
    filter.update()
    final_cell_density_image = filter.getFinalCellDensityImage()

    # Result writing
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(join(data_dir, 'final.mha'))
    writer.SetInputData(final_cell_density_image)
    writer.Write()

    # Plot
    figure_1 = plt.figure("Brain Domain")
    axis_1_1 = figure_1.add_subplot(121)
    axis_1_1.imshow(np.transpose(image_to_array(brain_map_image)[:, :, 191, 0]), cmap='gray')
    axis_1_2 = figure_1.add_subplot(222)
    axis_1_2.imshow(np.transpose(image_to_array(brain_map_image)[209, :, :, 0]), cmap='gray', origin='lower')
    axis_1_3 = figure_1.add_subplot(224)
    axis_1_3.imshow(np.transpose(image_to_array(brain_map_image)[:, 298, :, 0]), cmap='gray', origin='lower')

    figure_2 = plt.figure("Initial Tumour Cell Density")
    axis_2_1 = figure_2.add_subplot(121)
    axis_2_1.imshow(np.transpose(image_to_array(initial_cell_density_image)[:, :, 191, 0]), cmap='jet', vmin=0.0, vmax=1.0)
    axis_2_2 = figure_2.add_subplot(222)
    axis_2_2.imshow(np.transpose(image_to_array(initial_cell_density_image)[209, :, :, 0]), cmap='jet', vmin=0.0, vmax=1.0, origin='lower')
    axis_2_3 = figure_2.add_subplot(224)
    axis_2_3.imshow(np.transpose(image_to_array(initial_cell_density_image)[:, 298, :, 0]), cmap='jet', vmin=0.0, vmax=1.0, origin='lower')

    figure_3 = plt.figure("Final Tumour Cell Density")
    axis_3_1 = figure_3.add_subplot(121)
    axis_3_1.imshow(np.transpose(image_to_array(final_cell_density_image)[:, :, 191, 0]), cmap='jet', vmin=0.0, vmax=1.0)
    axis_3_2 = figure_3.add_subplot(222)
    axis_3_2.imshow(np.transpose(image_to_array(final_cell_density_image)[209, :, :, 0]), cmap='jet', vmin=0.0, vmax=1.0, origin='lower')
    axis_3_3 = figure_3.add_subplot(224)
    axis_3_3.imshow(np.transpose(image_to_array(final_cell_density_image)[:, 298, :, 0]), cmap='jet', vmin=0.0, vmax=1.0, origin='lower')

    plt.show()


if __name__ == '__main__':

    run_example(sys.argv[1])
